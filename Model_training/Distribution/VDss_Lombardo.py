# UPDATED VDss_Lombardo.py WITH REGRESSION METRICS (R², MAE, MSE)

import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from paths import train_path, valid_path, test_path, prediction_path, model_path

FP_SIZE = 2048

# ---------- Descriptor Builder ----------
def descriptors_from_mol(mol):
    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        int(Lipinski.NumHDonors(mol)),
        int(Lipinski.NumHAcceptors(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        int(mol.GetNumAtoms()),
    ]

gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def fingerprint_from_mol(mol):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((FP_SIZE,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ---------- Feature Builder ----------
def build_features(df, smiles_col="Drug", use_desc=True, use_fp=True):
    feat_rows, desc_rows, idxs = [], {}, []
    for idx, smi in df[smiles_col].astype(str).items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        parts = []

        if use_desc:
            d = descriptors_from_mol(mol)
            parts.append(np.array(d, dtype=float))
            desc_rows[idx] = d
        
        if use_fp:
            fp = fingerprint_from_mol(mol)
            parts.append(fp.astype(float))

        feat = np.concatenate(parts)
        feat_rows.append(feat)
        idxs.append(idx)

    X = np.vstack(feat_rows)
    desc_df = pd.DataFrame.from_dict(
        desc_rows, orient="index",
        columns=[
            "MolWt","LogP","NumHDonors","NumHAcceptors",
            "TPSA","NumRotatableBonds","AtomCount"
        ]
    )
    return X, desc_df, idxs

# ---------- Train Model ----------
def train_model(train_csv, valid_csv, smiles_col="Drug", label_col="Y",
                save_model_path=None, use_desc=True, use_fp=True,
                model_type="auto"):

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    X_train, desc_train, idx_train = build_features(train_df, smiles_col, use_desc, use_fp)
    X_valid, desc_valid, idx_valid = build_features(valid_df, smiles_col, use_desc, use_fp)

    y_train = train_df.loc[idx_train, label_col].values
    y_valid = valid_df.loc[idx_valid, label_col].values

    # ALWAYS REGRESSION FOR VDss
    is_classification = False
    print("Task detected: REGRESSION (VDss_Lombardo)")

    # ---- Select Model ----
    if model_type == "auto":
        model_type = "xgb"

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        )
    else:
        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

    print("Training model...")
    model.fit(X_train, y_train)

    # -------- Validation (Regression) --------
    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print("\nValidation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE     : {mae:.4f}")
    print(f"MSE     : {mse:.4f}")

    meta = {
        "model": model,
        "is_classification": False,
        "dataset_name": "VDss",
        "use_desc": use_desc,
        "use_fp": use_fp
    }

    if save_model_path:
        joblib.dump(meta, save_model_path)
        print(f"Model saved → {save_model_path}")

    return meta

# ---------- Evaluate & Save Predictions ----------
def evaluate_and_save_predictions(meta, test_csv, prediction_csv=None,
                                  smiles_col="Drug", label_col="Y"):

    df = pd.read_csv(test_csv)

    X_test, desc_df, idx_test = build_features(df, smiles_col,
                                               meta["use_desc"],
                                               meta["use_fp"])
    model = meta["model"]
    y_true = df.loc[idx_test, label_col].values

    print("\n===== VDss_Lombardo Test Evaluation (Regression) =====")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"MAE     : {mae:.4f}")
    print(f"MSE     : {mse:.4f}")

    # ------- Save Predictions -------
    df["Predicted_Value"] = np.nan
    for i, idx in enumerate(idx_test):
        df.at[idx, "Predicted_Value"] = y_pred[i]

    df["Units"] = "L/kg"

    if prediction_csv:
        df.to_csv(prediction_csv, index=False)
        print(f"Predictions saved → {prediction_csv}")

    return df

# ---------- Run ----------
if __name__ == "__main__":

    category = "Distribution"
    model_name = "VDss_Lombardo"

    train_csv = train_path(category, model_name)
    valid_csv = valid_path(category, model_name)
    test_csv  = test_path(category, model_name)
    save_file = model_path(category, model_name)
    pred_file = prediction_path(category, model_name)

    meta = train_model(train_csv, valid_csv, save_model_path=save_file)
    evaluate_and_save_predictions(meta, test_csv, pred_file)
