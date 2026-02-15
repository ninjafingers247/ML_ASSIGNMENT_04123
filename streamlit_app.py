import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier


st.set_page_config(page_title="Telco Churn — Streamlit App", layout="wide")

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"
DROP_COLS = ["customerID"]
RANDOM_STATE = 42

MODEL_DIR = "model"
ARTIFACT_PATH = os.path.join(MODEL_DIR, "models.joblib")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.csv")
METRICS_CSV_PATH = os.path.join(MODEL_DIR, "metrics.csv")


def compute_binary_auc(y_true_enc, proba_2d):
    return roc_auc_score(y_true_enc, proba_2d[:, 1])


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_telco_dataframe():
    df = pd.read_csv(DATA_PATH)

    # Telco-specific cleanup: TotalCharges has blank strings; convert to numeric
    if "TotalCharges" in df.columns:
        tc = df["TotalCharges"].astype(str).str.strip().replace({"": np.nan})
        df["TotalCharges"] = pd.to_numeric(tc, errors="coerce")

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}")

    return df


def pick_threshold(y_true, pos_proba):
    # Grid-search a threshold that maximizes F1; tie-break by MCC.
    best = {"thr": 0.5, "f1": -1.0, "mcc": -2.0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (pos_proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_true, pred)
        if (f1 > best["f1"]) or (f1 == best["f1"] and mcc > best["mcc"]):
            best = {"thr": float(thr), "f1": float(f1), "mcc": float(mcc)}
    return best["thr"]


def train_and_save():
    """
    Train all 6 required models, compute metrics, and save:
    - model/models.joblib
    - model/metrics.csv
    - model/test_data.csv
    """
    ensure_dirs()

    df = load_telco_dataframe()
    y_raw = df[TARGET_COL].astype(str)
    X = df.drop(columns=[TARGET_COL])

    # Infer types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Encode labels -> 0/1, with positive class = 1 (typically "Yes")
    y_le = LabelEncoder()
    y = y_le.fit_transform(y_raw)
    class_names = y_le.classes_.tolist()
    if len(class_names) != 2:
        raise ValueError(f"Expected binary target, got classes={class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Validation split for threshold selection
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Save test CSV for upload demo (contains true labels)
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_le.inverse_transform(y_test)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    # Baseline (majority class)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_fit, y_fit)
    dummy_pred = dummy.predict(X_test)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "k-Nearest Neighbor": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
        ),
    }

    results = []
    fitted = {}
    thresholds = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_fit, y_fit)

        # Threshold selection on validation
        val_proba = pipe.predict_proba(X_val)[:, 1]
        thr = pick_threshold(y_val, val_proba)
        thresholds[name] = thr

        # Test metrics
        proba = pipe.predict_proba(X_test)
        y_pred = (proba[:, 1] >= thr).astype(int)
        auc = compute_binary_auc(y_test, proba)

        results.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "auc": float(auc),
                "precision": float(
                    precision_score(y_test, y_pred, pos_label=1, zero_division=0)
                ),
                "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_test, y_pred)),
            }
        )
        fitted[name] = pipe

    metrics_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    artifact = {
        "data_path": DATA_PATH,
        "target_col": TARGET_COL,
        "drop_cols": DROP_COLS,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "class_names": class_names,
        "label_encoder": y_le,
        "models": fitted,
        "thresholds": thresholds,
        "dummy_baseline": {
            "accuracy": float(accuracy_score(y_test, dummy_pred)),
            "precision": float(precision_score(y_test, dummy_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_test, dummy_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_test, dummy_pred, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, dummy_pred)),
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    return metrics_df


def ensure_artifact_exists():
    """
    Streamlit Cloud runs the app directly and may not have a pre-generated artifact.
    If missing, allow training inside the app (writes models/ + reports/).
    """
    if os.path.exists(ARTIFACT_PATH):
        return

    st.warning(
        "Model artifact not found (`model/models.joblib`). "
        "Click the button below to train models in this environment."
    )
    if st.button("Train models now (creates models/ and reports/)"):
        with st.spinner("Training models… this may take 1–2 minutes."):
            train_and_save()
        # Clear cached resources and reload.
        st.cache_resource.clear()
        st.success("Training complete. Reloading…")
        st.rerun()

    st.stop()


def main():
    st.title("Telco Customer Churn — Model Comparator")

    ensure_artifact_exists()
    artifact = load_artifact()
    models = artifact["models"]
    target_col = artifact["target_col"]
    drop_cols = artifact.get("drop_cols", [])
    class_names = artifact["class_names"]
    y_le = artifact["label_encoder"]
    thresholds = artifact.get("thresholds", {})

    with st.sidebar:
        st.header("Inputs")
        model_name = st.selectbox("Choose a model", list(models.keys()))
        uploaded = st.file_uploader("Upload CSV (test data)", type=["csv"])
        run = st.button("Run prediction")

    st.caption(
        f"Target: `{target_col}` | Dropped columns if present: {drop_cols} | Classes: {class_names}"
    )

    if not uploaded:
        st.info(
            "Upload a CSV to begin. Tip: after training, upload `model/test_data.csv` "
            "or download it below."
        )
        if os.path.exists(TEST_DATA_PATH):
            with open(TEST_DATA_PATH, "rb") as f:
                st.download_button(
                    "Download model/test_data.csv",
                    data=f.read(),
                    file_name="test_data.csv",
                    mime="text/csv",
                )
        if os.path.exists(METRICS_CSV_PATH):
            st.caption("Latest saved metrics:")
            st.dataframe(pd.read_csv(METRICS_CSV_PATH), use_container_width=True)
        return

    df = pd.read_csv(uploaded)
    
    # Apply the same TotalCharges preprocessing as in training
    if "TotalCharges" in df.columns:
        tc = df["TotalCharges"].astype(str).str.strip().replace({"": np.nan})
        df["TotalCharges"] = pd.to_numeric(tc, errors="coerce")
    
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if not run:
        return

    df2 = df.copy()
    for c in drop_cols:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])

    y_true_str = None
    if target_col in df2.columns:
        y_true_str = df2[target_col].astype(str)
        df2 = df2.drop(columns=[target_col])

    model = models[model_name]
    thr = thresholds.get(model_name, 0.5)

    # Predict
    proba = model.predict_proba(df2)
    y_pred_enc = (proba[:, 1] >= thr).astype(int)
    y_pred_str = y_le.inverse_transform(y_pred_enc)

    out = df.copy()
    out["prediction"] = y_pred_str

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Predictions")
        st.dataframe(out.head(25), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("Evaluation (only if true labels are included)")

        if y_true_str is None:
            st.warning(f"No `{target_col}` column found in upload → cannot compute metrics.")
            return

        # Validate labels
        unknown = sorted(set(y_true_str.unique()) - set(y_le.classes_))
        if unknown:
            st.error(f"Upload contains unknown labels not seen in training: {unknown}")
            return

        y_true_enc = y_le.transform(y_true_str)

        auc = compute_binary_auc(y_true_enc, proba)

        metrics = {
            "accuracy": float(accuracy_score(y_true_enc, y_pred_enc)),
            "auc": float(auc),
            "precision": float(precision_score(y_true_enc, y_pred_enc, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true_enc, y_pred_enc, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true_enc, y_pred_enc, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true_enc, y_pred_enc)),
        }

        st.table(pd.DataFrame(metrics, index=[model_name]).T)
        st.caption(f"Decision threshold used for positive class: {thr:.2f}")

        cm = confusion_matrix(y_true_enc, y_pred_enc)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig)


if __name__ == "__main__":
    main()


