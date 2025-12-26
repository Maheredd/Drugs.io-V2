# UPDATED BBB_Martins.py WITH ACCURACY, F1, CONFUSION MATRIX, ROC–AUC

import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    f1_score, accuracy_score,
    mean_squared_error, r2_score
)
from paths import train_path, valid_path, test_path, prediction_path, model_path

FP_SIZE = 2048

# ---------- Feature Builders ----------
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

gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def fingerprint_from_mol(mol):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((FP_SIZE,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_features_and_desc(df, smiles_col="Drug", use_descriptors=True, use_fingerprints=True):
    feats, desc_rows, valid_idx = [], {}, []
    for idx, smi in df[smiles_col].astype(str).items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        parts = []
        if use_descriptors:
            d = descriptors_from_mol(mol)
            parts.append(np.array(d, dtype=float))
            desc_rows[idx] = d

        if use_fingerprints:
            arr = fingerprint_from_mol(mol)
            parts.append(arr.astype(float))

        feat = np.concatenate(parts)
        feats.append(feat)
        valid_idx.append(idx)

    X = np.vstack(feats)
    desc_df = pd.DataFrame.from_dict(
        desc_rows, orient="index",
        columns=["MolWt","LogP","NumHDonors","NumHAcceptors","TPSA",
                 "NumRotatableBonds","AtomCount"]
    )
    return X, desc_df, valid_idx

# ---------- Druglikeness ----------
def druglike(mol):
    if mol is None:
        return ["Invalid"] * 5
    L = (Descriptors.MolWt(mol) <= 500 and Crippen.MolLogP(mol) <= 5)
    G = (160 <= Descriptors.MolWt(mol) <= 480)
    V = (rdMolDescriptors.CalcTPSA(mol) <= 140)
    E = (rdMolDescriptors.CalcTPSA(mol) <= 131)
    M = (200 <= Descriptors.MolWt(mol) <= 600)
    return ["Yes" if x else "No" for x in [L, G, V, E, M]]

ADMET_UNITS = {"BBB": "-"}

# ---------- Train ----------
def train_model(train_csv, valid_csv, smiles_col="Drug", label_col="Y",
                save_model_path=None, use_descriptors=True,
                use_fingerprints=True, model_type="auto"):

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    X_train, _, idx_train = build_features_and_desc(train_df, smiles_col)
    y_train = train_df.loc[idx_train, label_col].values

    X_valid, _, idx_valid = build_features_and_desc(valid_df, smiles_col)
    y_valid = valid_df.loc[idx_valid, label_col].values

    # BBB is ALWAYS classification
    is_classification = True
    y_train = y_train.astype(int)
    y_valid = y_valid.astype(int)

    print("Task detected: CLASSIFICATION (BBB Model)")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Validation
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    print("\nValidation Results:")
    print(classification_report(y_valid, y_pred))

    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average="weighted")
    cm = confusion_matrix(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"ROC-AUC: {auc:.4f}")

    meta = {
        "model": model,
        "is_classification": True,
        "dataset_name": "BBB"
    }

    if save_model_path:
        joblib.dump(meta, save_model_path)
        print(f"Model saved → {save_model_path}")

    return meta

# ---------- Evaluate & Predict ----------
def evaluate_and_save_predictions(model_meta, test_csv, prediction_csv=None,
                                  smiles_col="Drug", label_col="Y"):

    df = pd.read_csv(test_csv)

    X_test, desc_df, idx_test = build_features_and_desc(df, smiles_col)
    model = model_meta["model"]

    y_true = df.loc[idx_test, label_col].values.astype(int)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== BBB_Martins Test Set Evaluation =====")
    print(classification_report(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"ROC-AUC: {auc:.4f}")

    df["Predicted_Class"] = np.nan
    df["Prob_Class_0"] = np.nan
    df["Prob_Class_1"] = np.nan

    for i, idx in enumerate(idx_test):
        df.at[idx, "Predicted_Class"] = y_pred[i]
        df.at[idx, "Prob_Class_0"] = 1 - y_prob[i]
        df.at[idx, "Prob_Class_1"] = y_prob[i]

    df["Units"] = ADMET_UNITS["BBB"]

    if prediction_csv:
        df.to_csv(prediction_csv, index=False)
        print(f"Predictions saved → {prediction_csv}")

    return df

# ---------- Run ----------
if __name__ == "__main__":
    category = "Distribution"
    model_name = "BBB_Martins"

    train_csv = train_path(category, model_name)
    valid_csv = valid_path(category, model_name)
    test_csv  = test_path(category, model_name)
    save_model = model_path(category, model_name)
    pred_csv   = prediction_path(category, model_name)

    meta = train_model(train_csv, valid_csv, save_model_path=save_model)
    evaluate_and_save_predictions(meta, test_csv, pred_csv)
