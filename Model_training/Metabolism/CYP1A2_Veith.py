# CYP1A2_Veith.py (UPDATED WITH FULL TRAIN + TEST METRICS)
# Includes Accuracy, F1, Confusion Matrix, ROC-AUC, Classification Report

import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, Lipinski
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from paths import train_path, valid_path, test_path, prediction_path, model_path

# ---------- Fingerprint size ----------
FP_SIZE = 2048

# ---------- Feature builders ----------
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
    feats = []
    desc_rows = {}
    valid_idx = []

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
        columns=["MolWt","LogP","NumHDonors","NumHAcceptors","TPSA","NumRotatableBonds","AtomCount"]
    ) if use_descriptors else None

    return X, desc_df, valid_idx

# ---------- Druglikeness rules ----------
def lipinski_rule(mol):
    return (Descriptors.MolWt(mol) <= 500 and Crippen.MolLogP(mol) <= 5 and
            Lipinski.NumHDonors(mol) <= 5 and Lipinski.NumHAcceptors(mol) <= 10)

def ghose_rule(mol):
    mw = Descriptors.MolWt(mol); logp = Crippen.MolLogP(mol); atoms = mol.GetNumAtoms()
    return (160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 20 <= atoms <= 70)

def veber_rule(mol):
    return (rdMolDescriptors.CalcTPSA(mol) <= 140 and rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10)

def egan_rule(mol):
    return (rdMolDescriptors.CalcTPSA(mol) <= 131 and Crippen.MolLogP(mol) <= 5.88)

def muegge_rule(mol):
    mw = Descriptors.MolWt(mol); logp = Crippen.MolLogP(mol)
    return (200 <= mw <= 600 and -2 <= logp <= 5 and Lipinski.NumHAcceptors(mol) <= 10 and
            Lipinski.NumHDonors(mol) <= 5 and rdMolDescriptors.CalcTPSA(mol) <= 150)

# ---------- Train model ----------
def train_model(train_csv, valid_csv, smiles_col="Drug", label_col="Y",
                save_model_path=None, use_descriptors=True, use_fingerprints=True,
                model_type="xgb"):

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    X_train, desc_train, idx_train = build_features_and_desc(train_df, smiles_col,
                                                             use_descriptors, use_fingerprints)
    y_train = train_df.loc[idx_train, label_col].astype(int).values

    X_valid, desc_valid, idx_valid = build_features_and_desc(valid_df, smiles_col,
                                                             use_descriptors, use_fingerprints)
    y_valid = valid_df.loc[idx_valid, label_col].astype(int).values

    pos, neg = np.sum(y_train==1), np.sum(y_train==0)

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        scale_pos_weight = float(neg) / max(1.0, float(pos))
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)

    # ---------- TRAINING METRICS ----------
    if X_valid.shape[0] > 0:
        yv_pred = model.predict(X_valid)

        try:
            yv_prob = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, yv_prob)
        except:
            yv_prob = None
            auc = None

        print("\n=== TRAINING VALIDATION METRICS ===")
        print(classification_report(y_valid, yv_pred))

        print("Accuracy:", accuracy_score(y_valid, yv_pred))
        print("F1 Score (weighted):", f1_score(y_valid, yv_pred, average="weighted"))
        print("Confusion Matrix:\n", confusion_matrix(y_valid, yv_pred))
        print("ROC-AUC:", auc)

    # Save model package
    meta = {
        "model": model,
        "use_descriptors": use_descriptors,
        "use_fingerprints": use_fingerprints
    }

    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(meta, save_model_path)
        print("\nModel saved:", save_model_path)

    return meta

# ---------- Evaluate + Save Predictions ----------
def evaluate_and_save_predictions(meta, test_csv, prediction_csv,
                                  smiles_col="Drug", label_col="Y"):

    df_test = pd.read_csv(test_csv)

    X_test, desc_test, idx_test = build_features_and_desc(df_test, smiles_col,
                                                          meta["use_descriptors"],
                                                          meta["use_fingerprints"])

    model = meta["model"]

    # ---------- TEST METRICS ----------
    print("\n=== TEST SET METRICS ===")

    y_true = df_test.loc[idx_test, label_col].astype(int).values
    y_pred = model.predict(X_test)

    try:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_true, probs)
    except:
        probs = None
        auc = None

    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score (weighted):", f1_score(y_true, y_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ROC-AUC:", auc)

    # ---------- SAVE PREDICTIONS ----------
    result = df_test.copy()
    result["Predicted_Class"] = np.nan
    result["Prob_Class_0"] = np.nan
    result["Prob_Class_1"] = np.nan

    for i, idx in enumerate(idx_test):
        result.at[idx, "Predicted_Class"] = y_pred[i]
        if probs is not None:
            result.at[idx, "Prob_Class_1"] = probs[i]
            result.at[idx, "Prob_Class_0"] = 1 - probs[i]

    # Druglikeness rules
    for rule_name, rule_func in zip(
        ["Lipinski", "Ghose", "Veber", "Egan", "Muegge"],
        [lipinski_rule, ghose_rule, veber_rule, egan_rule, muegge_rule]):
        
        result[rule_name] = [
            "Invalid" if Chem.MolFromSmiles(s) is None
            else ("Yes" if rule_func(Chem.MolFromSmiles(s)) else "No")
            for s in result[smiles_col].astype(str)
        ]

    result["Units"] = "-"

    if prediction_csv:
        os.makedirs(os.path.dirname(prediction_csv), exist_ok=True)
        result.to_csv(prediction_csv, index=False)
        print("\nPredictions saved to:", prediction_csv)

    return result

# ---------- MAIN ----------
if __name__ == "__main__":
    category = "Metabolism"
    model_name = "CYP1A2_Veith"

    train_csv = train_path(category, model_name)
    valid_csv = valid_path(category, model_name)
    test_csv  = test_path(category, model_name)
    save_file = model_path(category, model_name)
    pred_file = prediction_path(category, model_name)

    meta = train_model(train_csv, valid_csv,
                       use_descriptors=True,
                       use_fingerprints=True,
                       model_type="xgb",
                       save_model_path=save_file)

    evaluate_and_save_predictions(meta, test_csv, pred_file)
