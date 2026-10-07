import numpy as np
import pandas as pd
import torch
import joblib
from model import IntrusionDNN

def load_system(joblib_path, model_path):
    bundle = joblib.load(joblib_path)
    
    input_size = len(bundle["feature_names"])
    num_classes = len(bundle["label_encoder"].classes_)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntrusionDNN(input_size, num_classes, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return bundle, model, device

def preprocess_live_traffic(df, bundle):
    # 1. Keep only raw feature cols needed
    X = df.copy()
    available_cols = [c for c in bundle["raw_feature_cols"] if c in X.columns]
    X = X[available_cols].astype(float)
    
    # 2. Window Features
    if "Init_Win_bytes_forward" in X.columns:
        X["win_fwd_missing"] = (X["Init_Win_bytes_forward"] < 0).astype(float)
        X["Init_Win_bytes_forward"] = X["Init_Win_bytes_forward"].clip(lower=0)
    if "Init_Win_bytes_backward" in X.columns:
        X["win_bwd_missing"] = (X["Init_Win_bytes_backward"] < 0).astype(float)
        X["Init_Win_bytes_backward"] = X["Init_Win_bytes_backward"].clip(lower=0)
        
    # 3. Clip Negatives
    for c in X.columns:
        if c not in ("win_fwd_missing", "win_bwd_missing"):
            X[c] = X[c].clip(lower=0)
            
    # 4. Protocol OHE
    if "Protocol" in X.columns:
        proto = X.pop("Protocol").round().astype(int)
        X["proto_tcp"] = (proto == 6).astype(float)
        X["proto_udp"] = (proto == 17).astype(float)
        X["proto_other"] = (~proto.isin([6, 17])).astype(float)
        
    if bundle.get("drop_window", False):
        WINDOW_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward", "win_fwd_missing", "win_bwd_missing"]
        X = X.drop(columns=[c for c in WINDOW_COLS if c in X.columns])
        
    # Ensure exact feature order
    for missing_col in bundle["feature_names"]:
        if missing_col not in X.columns:
            X[missing_col] = 0.0
            
    X_out = X[bundle["feature_names"]].copy()
    
    # Scale
    cont = bundle["continuous_cols"]
    X_out[cont] = np.log1p(X_out[cont])
    X_out[cont] = bundle["scaler"].transform(X_out[cont])
    
    return X_out.to_numpy(dtype="float32")

def predict(X_numpy, bundle, model, device, conf_threshold=0.80):
    tensor_X = torch.tensor(X_numpy, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor_X)
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
    
    max_probs = max_probs.cpu().numpy()
    preds = preds.cpu().numpy()
    
    classes = bundle["label_encoder"].classes_
    labels = np.array([classes[p] for p in preds])
    
    # Apply Confidence Threshold (Claude's recommendation)
    low_conf = max_probs < conf_threshold
    labels[low_conf] = "Uncertain / Unrecognized"
    
    return labels, max_probs
