import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix
)
import joblib
import numpy as np
from paths import train_path, valid_path, test_path, prediction_path, model_path

# ---------- Helper: Compute Molecular Descriptors ----------
def compute_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "NumHDonors": int(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumRotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "AtomCount": int(mol.GetNumAtoms())
    }

# ---------- Druglikeness Evaluation ----------
def evaluate_druglikeness_from_desc(desc: dict) -> dict:
    if desc is None:
        return {"Lipinski": "Invalid", "Ghose": "Invalid", "Veber": "Invalid",
                "Egan": "Invalid", "Muegge": "Invalid"}

    violations = sum([
        desc["MolWt"] > 500,
        desc["LogP"] > 5,
        desc["NumHDonors"] > 5,
        desc["NumHAcceptors"] > 10
    ])

    lipinski = "Yes" if violations <= 1 else "No"
    ghose = "Yes" if (160 <= desc["MolWt"] <= 480 and -0.4 <= desc["LogP"]
                      <= 5.6 and 40 <= desc["AtomCount"] <= 70) else "No"
    veber = "Yes" if (desc["NumRotatableBonds"] <= 10 and desc["TPSA"] <= 140) else "No"
    egan = "Yes" if (desc["LogP"] <= 5.88 and desc["TPSA"] <= 131) else "No"
    muegge = "Yes" if (200 <= desc["MolWt"] <= 600 and desc["LogP"] <= 5 and
                       desc["NumRotatableBonds"] <= 15 and desc["TPSA"] <= 150) else "No"
    return {"Lipinski": lipinski, "Ghose": ghose, "Veber": veber,
            "Egan": egan, "Muegge": muegge}

# ---------- Auto-detect task type ----------
def detect_task_type(df: pd.DataFrame, target_col="Y"):
    return "classification", "-"

# ---------- Train Model ----------
def train_model(train_csv: str, smiles_col="Drug", label_col="Y",
                save_model_path=None):

    df = pd.read_csv(train_csv)
    task_type, units = detect_task_type(df, label_col)

    desc_list = [compute_descriptors(s) for s in df[smiles_col].astype(str)]
    desc_df = pd.DataFrame(desc_list)
    df = pd.concat([df, desc_df], axis=1).dropna()

    feature_cols = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors",
                    "TPSA", "NumRotatableBonds", "AtomCount"]

    X = df[feature_cols].values
    y = df[label_col].values.astype(int)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "task_type": task_type, "units": units},
                save_model_path)

    print(f"Model trained and saved to {save_model_path}")
    return model, feature_cols, task_type, units

# ---------- Evaluate Model ----------
def evaluate_model(model, feature_cols, test_csv,
                   smiles_col="Drug", label_col="Y",
                   task_type="classification"):

    df = pd.read_csv(test_csv)
    desc_list = [compute_descriptors(s) for s in df[smiles_col].astype(str)]
    desc_df = pd.DataFrame(desc_list)
    df = pd.concat([df, desc_df], axis=1).dropna()

    X = df[feature_cols].values
    y = df[label_col].values.astype(int)

    if task_type == "classification":
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        print("\n==== HIA_Hou Classification Metrics ====")
        print(classification_report(y, y_pred))

        # NEW METRICS
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        cm = confusion_matrix(y, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        try:
            auc = roc_auc_score(y, y_prob[:, 1])
            print(f"ROC-AUC: {auc:.4f}")
        except:
            print("ROC-AUC could not be computed")

    else:
        print("This model is classification only.")

# ---------- Predict and Save ----------
def predict_and_save_binary(model, feature_cols, valid_csv, out_csv,
                            smiles_col="Drug", task_type="classification",
                            units="-"):

    df = pd.read_csv(valid_csv)
    desc_list = [compute_descriptors(s) for s in df[smiles_col].astype(str)]
    desc_df = pd.DataFrame(desc_list)
    df = pd.concat([df, desc_df], axis=1).dropna()

    X = df[feature_cols].values

    probs = model.predict_proba(X)
    df["Prob_Class_0"] = probs[:, 0]
    df["Prob_Class_1"] = probs[:, 1]
    df["Predicted_Class"] = model.predict(X)

    druglike = df[smiles_col].astype(str).apply(
        lambda s: evaluate_druglikeness_from_desc(compute_descriptors(s))
    )
    df = pd.concat([df, pd.DataFrame(druglike.tolist(), index=df.index)], axis=1)

    df["Units"] = units
    df.to_csv(out_csv, index=False)

    print(f"Predictions saved to {out_csv}")
    return df

# ---------- Example Usage ----------
if __name__ == "__main__":

    category = "Absorption"
    model_name = "HIA_Hou"

    train_csv = train_path(category, model_name)
    valid_csv = valid_path(category, model_name)
    test_csv = test_path(category, model_name)
    out_csv = prediction_path(category, model_name)
    save_model_file = model_path(category, model_name)

    model, feature_cols, task_type, units = train_model(
        train_csv, save_model_path=save_model_file
    )

    evaluate_model(model, feature_cols, test_csv, task_type=task_type)

    df_valid = predict_and_save_binary(
        model, feature_cols, valid_csv, out_csv,
        task_type=task_type, units=units
    )

    print(df_valid.head())
