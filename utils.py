import numpy as np
import pickle, json, os
from sklearn.linear_model import LinearRegression

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def inv_y(y_scaled, y_scaler):
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled).ravel()

def to_price(y_label, prev_close, label_type):
    y_label = np.asarray(y_label).ravel()
    if label_type == "price":
        return y_label
    prev_close = np.asarray(prev_close).ravel()
    if label_type == "log_return":
        return prev_close * np.exp(y_label)      # P_{t+1} = P_t * e^r
    if label_type == "return":
        return prev_close * (1.0 + y_label)     # P_{t+1} = P_t * (1+r)
    raise ValueError("label_type must be 'price' | 'return' | 'log_return'")

def reg_metrics_price(y_true_price, y_pred_price, prev_close=None):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true_price = np.asarray(y_true_price).ravel()
    y_pred_price = np.asarray(y_pred_price).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))
    mae  = float(np.mean(np.abs(y_true_price - y_pred_price)))
    r2   = float(r2_score(y_true_price, y_pred_price))
    smape = 100.0 * np.mean(
        2.0 * np.abs(y_pred_price - y_true_price) / np.clip(np.abs(y_true_price) + np.abs(y_pred_price), 1e-9, None)
    )
    da = None
    if prev_close is not None:
        prev_close = np.asarray(prev_close).ravel()
        da = float(np.mean(np.sign(y_true_price - prev_close) == np.sign(y_pred_price - prev_close)))
    return {"RMSE_USD": rmse, "MAE_USD": mae, "R2": r2, "sMAPE%": smape, "DA": da}

def maybe_calibrate(val_pred_price, val_true_price):
    lr = LinearRegression().fit(val_pred_price.reshape(-1,1), val_true_price)
    return lr

def inverse_mae_weights(mae_dict):
    # convert MAE -> weight ~ 1/MAE and normalize
    w = {k: 1.0/max(1e-9, float(v)) for k,v in mae_dict.items()}
    s = sum(w.values())
    return {k: v/s for k,v in w.items()}

def weighted_ensemble(preds_by_view, weights):
    keys = list(preds_by_view.keys())
    P = np.vstack([np.asarray(preds_by_view[k]).ravel() for k in keys])  # (V, N)
    w = np.array([weights[k] for k in keys]).reshape(-1,1)               # (V, 1)
    return (P * w).sum(axis=0)
