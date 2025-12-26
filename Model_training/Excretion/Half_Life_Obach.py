# UPDATED Half_Life_Obach.py WITH REGRESSION METRICS (R², MAE, MSE)

import os
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from paths import train_path, valid_path, test_path, prediction_path, model_path


FP_SIZE = 2048
gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)

# ---------- DESCRIPTOR + FINGERPRINT BUILDERS ----------

def descriptors_from_mol(mol):
    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        int(Lipinski.NumHDonors(mol)),
        int(Lipinski.NumHAcceptors(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        int(mol.GetNumAtoms())
    ]


def fingerprint_from_mol(mol):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((FP_SIZE,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df, smiles_col="Drug", use_desc=True, use_fp=True):
    rows, idxs = [], []
    for idx, smi in df[smiles_col].astype(str).items():

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        parts = []
        if use_desc:
            parts.append(np.array(descriptors_from_mol(mol), dtype=float))
        if use_fp:
            parts.append(fingerprint_from_mol(mol).astype(float))

        rows.append(np.concatenate(parts))
        idxs.append(idx)

    X = np.vstack(rows)
    return X, idxs


# ---------- TRAIN MODEL ----------

def train_model(train_csv, valid_csv,
                smiles_col="Drug",
                label_col="Y",
                save_model_path=None):

    print("Detected task: REGRESSION (Half-Life Obach)")

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    X_train, idx_train = build_features(train_df, smiles_col)
    X_valid, idx_valid = build_features(valid_df, smiles_col)

    y_train = train_df.loc[idx_train, label_col].values
    y_valid = valid_df.loc[idx_valid, label_col].values

    # XGBoost Regressor
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBRegressor...")
    model.fit(X_train, y_train)

    # ---------- Validation ----------
    y_pred = model.predict(X_valid)

    print("\nValidation Metrics:")
    print(f"R² Score: {r2_score(y_valid, y_pred):.4f}")
    print(f"MAE     : {mean_absolute_error(y_valid, y_pred):.4f}")
    print(f"MSE     : {mean_squared_error(y_valid, y_pred):.4f}")

    meta = {
        "model": model,
        "is_classification": False,
        "use_descriptors": True,
        "use_fingerprints": True
    }

    if save_model_path:
        joblib.dump(meta, save_model_path)
        print(f"Model saved → {save_model_path}")

    return meta


# ---------- EVALUATE & SAVE PREDICTIONS ----------

def evaluate_and_save_predictions(meta, test_csv,
                                  prediction_csv=None,
                                  smiles_col="Drug",
                                  label_col="Y"):

    df = pd.read_csv(test_csv)

    X_test, idx_test = build_features(
        df, smiles_col,
        meta["use_descriptors"],
        meta["use_fingerprints"]
    )

    y_true = df.loc[idx_test, label_col].values
    model = meta["model"]

    print("\n===== Half_Life_Obach Test Evaluation (Regression) =====")

    y_pred = model.predict(X_test)

    print(f"R² Score : {r2_score(y_true, y_pred):.4f}")
    print(f"MAE      : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE      : {mean_squared_error(y_true, y_pred):.4f}")

    # Save predictions inside dataframe
    df["Predicted_HalfLife"] = np.nan
    for i, idx in enumerate(idx_test):
        df.at[idx, "Predicted_HalfLife"] = y_pred[i]

    df["Units"] = "hours"

    if prediction_csv:
        df.to_csv(prediction_csv, index=False)
        print(f"Predictions saved → {prediction_csv}")

    return df


# ---------- MAIN EXECUTION ----------

if __name__ == "__main__":

    category = "Excretion"
    model_name = "Half_Life_Obach"

    train_csv = train_path(category, model_name)
    valid_csv = valid_path(category, model_name)
    test_csv  = test_path(category, model_name)
    save_file = model_path(category, model_name)
    pred_file = prediction_path(category, model_name)

    meta = train_model(train_csv, valid_csv, save_model_path=save_file)
    evaluate_and_save_predictions(meta, test_csv, pred_file)
