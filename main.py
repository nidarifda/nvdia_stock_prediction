from fastapi import FastAPI, HTTPException
from model_server import PredictRequest, PredictResponse, _predict_view, LABEL_TYPE, val_mae
from utils import inverse_mae_weights, weighted_ensemble
import numpy as np

app = FastAPI(title="NVDA Price Service", version="1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label_type = req.label_type or LABEL_TYPE

    per_view = {}
    provided = list(req.views.keys())
    if not provided:
        raise HTTPException(status_code=400, detail="No views provided. Use keys like 'A','B','AFF'.")

    # per-view predictions
    for tag, payload in req.views.items():
        try:
            p_price, p_ret, used_cal = _predict_view(tag, payload, label_type, req.calibrate)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{tag}: {e}")
        per_view[tag] = dict(pred_price=p_price, pred_return=p_ret, used_calibration=used_cal)

    # optional ensemble
    ens = None
    weights = None
    if req.ensemble and len(per_view) > 1:
        # decide weights
        if req.weights:
            # user-provided weights (must sum to ~1)
            w = req.weights
            s = sum(w.values())
            weights = {k: float(v)/s for k,v in w.items() if k in per_view}
        else:
            # inverse MAE from validation (shipped in val_mae.json)
            mae = {k: val_mae[k] for k in per_view.keys() if k in val_mae}
            weights = inverse_mae_weights(mae)

        # align lengths, ensemble on price
        preds = {k: np.asarray(v["pred_price"], dtype=float) for k, v in per_view.items()}
        n = min(len(x) for x in preds.values())
        preds = {k: v[:n] for k,v in preds.items()}
        ens_price = weighted_ensemble(preds, weights)
        # compute return vs first view's prev_close if provided
        first_tag = next(iter(per_view))
        # we didn't retain P_t in the response, so return only price + zeros for return
        ens = dict(pred_price=ens_price.tolist(),
                   pred_return=[0.0]*len(ens_price),
                   used_calibration=any(per_view[k]["used_calibration"] for k in per_view))

    return PredictResponse(per_view=per_view, ensemble=ens, weights=weights, label_type=label_type)
