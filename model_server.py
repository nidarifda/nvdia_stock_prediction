import os, numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from soft_attention import SoftAttention
from utils import inv_y, to_price, load_pickle, load_json, maybe_calibrate, inverse_mae_weights, weighted_ensemble

ART_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
LABEL_TYPE = os.getenv("LABEL_TYPE")  # override via env if desired

# -----------------
# load artifacts
# -----------------
y_scaler = load_pickle(os.path.join(ART_DIR, "y_scaler.pkl"))
val_mae  = load_json(os.path.join(ART_DIR, "val_mae.json"))
meta     = load_json(os.path.join(ART_DIR, "meta.json")) if os.path.exists(os.path.join(ART_DIR,"meta.json")) else {}
if LABEL_TYPE is None:
    LABEL_TYPE = meta.get("label_type", "log_return")

# optional: AFF model if you add it later
MODELS = {}
for tag in ["A", "B", "AFF"]:
    path = os.path.join(ART_DIR, f"nvda_BiLSTM_Attn_{tag}_reg_optuna.keras")
    if os.path.exists(path):
        MODELS[tag] = load_model(path, custom_objects={"SoftAttention": SoftAttention})

# -----------------
# request/response schemas
# -----------------
class ViewPayload(BaseModel):
    X: List[List[List[float]]] = Field(..., description="3D windows: (N, T, F)")
    prev_close: Optional[List[float]] = Field(None, description="P_t per sample; required if LABEL_TYPE != price")
    # optional calibration split
    X_val: Optional[List[List[List[float]]]] = None
    y_val_sc: Optional[List[float]] = None
    prev_close_val: Optional[List[float]] = None

class PredictRequest(BaseModel):
    views: Dict[str, ViewPayload]
    label_type: Optional[str] = None
    calibrate: bool = False
    ensemble: bool = True
    # override weights: {"A": 0.5, "B": 0.5, ...}
    weights: Optional[Dict[str, float]] = None

class ViewPrediction(BaseModel):
    pred_price: List[float]
    pred_return: List[float]  # predicted (P_{t+1}-P_t)/P_t
    used_calibration: bool

class PredictResponse(BaseModel):
    per_view: Dict[str, ViewPrediction]
    ensemble: Optional[ViewPrediction] = None
    weights: Optional[Dict[str, float]] = None
    label_type: str

# -----------------
# core inference
# -----------------
def _predict_view(tag: str, payload: ViewPayload, label_type: str, calibrate: bool):
    if tag not in MODELS:
        raise ValueError(f"Model for view {tag} not loaded.")
    m = MODELS[tag]
    X = np.asarray(payload.X, dtype=np.float32)
    if label_type != "price":
        if payload.prev_close is None:
            raise ValueError(f"prev_close required for view {tag} when label_type={label_type}.")
        P = np.asarray(payload.prev_close, dtype=np.float32)
    else:
        P = None

    # predict in label space
    yhat_sc = m.predict(X, verbose=0).ravel()
    yhat_lbl = inv_y(yhat_sc, y_scaler)

    # convert to price
    pred_price = to_price(yhat_lbl, P, label_type) if label_type != "price" else yhat_lbl

    used_calib = False
    if calibrate and (payload.X_val is not None) and (payload.y_val_sc is not None):
        Xv = np.asarray(payload.X_val, dtype=np.float32)
        yv_sc = np.asarray(payload.y_val_sc, dtype=np.float32).ravel()
        yv_lbl = inv_y(yv_sc, y_scaler)
        if label_type != "price":
            Pv = np.asarray(payload.prev_close_val, dtype=np.float32)
            pred_val_sc  = m.predict(Xv, verbose=0).ravel()
            pred_val_lbl = inv_y(pred_val_sc, y_scaler)
            pred_val_price = to_price(pred_val_lbl, Pv, label_type)
            true_val_price = to_price(yv_lbl, Pv, label_type)
        else:
            pred_val_price = m.predict(Xv, verbose=0).ravel()  # already price labels (scaled->lbl)
            pred_val_price = inv_y(pred_val_price, y_scaler)
            true_val_price = yv_lbl
        lr_head = maybe_calibrate(pred_val_price.reshape(-1,1), true_val_price)
        pred_price = lr_head.predict(pred_price.reshape(-1,1)).ravel()
        used_calib = True

    # also return predicted return vs P_t if available
    if P is None:
        pred_ret = np.zeros_like(pred_price, dtype=float).tolist()
    else:
        pred_ret = ((pred_price - P) / np.clip(P, 1e-12, None)).tolist()

    return pred_price.tolist(), pred_ret, used_calib
