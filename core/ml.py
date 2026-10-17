"""Machine-learning studio: AutoML-style model comparison, clustering,
anomaly detection and PCA, all built on scikit-learn pipelines so that
categorical columns, missing values and scaling are handled automatically.
"""
from __future__ import annotations

import io
import pickle
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .profiling import numeric_cols

REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.01, max_iter=5000),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
    "K-Nearest Neighbours": lambda: KNeighborsRegressor(n_neighbors=7),
    "Support Vector (SVR)": lambda: SVR(),
}
CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbours": lambda: KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": lambda: GaussianNB(),
    "Support Vector (SVC)": lambda: SVC(probability=True, random_state=42),
}
DEFAULT_REG = ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"]
DEFAULT_CLF = ["Logistic Regression", "Random Forest", "Gradient Boosting", "K-Nearest Neighbours"]


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = numeric_cols(X)
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=25, sparse_output=False))]), cat),
    ], remainder="drop", verbose_feature_names_out=False)


def prepare_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Make model inputs consistent: dates → day number, booleans → 0/1, everything else non-numeric → text."""
    X = df[features].copy()
    for c in features:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].map(lambda v: v.toordinal() if pd.notna(v) else np.nan).astype(float)
        elif pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    return X


def _prep_xy(df, target, features):
    data = df.dropna(subset=[target])
    return prepare_features(data, features), data[target]


def predict_frame(result, df: pd.DataFrame) -> np.ndarray:
    """Score a whole DataFrame with the best model (labels decoded for classification)."""
    pred = result["best_model"].predict(prepare_features(df, result["features"]))
    if result["task"] == "classification":
        pred = result["label_encoder"].inverse_transform(np.asarray(pred).astype(int))
    return pred


def train_and_compare(df, target, features, task=None, models=None, test_size=0.2, cv_folds=5, random_state=42):
    X, y = _prep_xy(df, target, features)
    task = task or infer_task(y)
    label_encoder = None
    if task == "classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError("Target needs at least two classes.")
        label_encoder = LabelEncoder().fit(y)
        y_enc = pd.Series(label_encoder.transform(y), index=y.index)
        counts = y_enc.value_counts()
        stratify = y_enc if counts.min() >= 2 else None
    else:
        y_enc = y.astype(float)
        stratify = None
    if len(X) < 20:
        raise ValueError("Need at least 20 rows with a non-missing target to train models.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=stratify)
    registry = REGRESSORS if task == "regression" else CLASSIFIERS
    models = models or (DEFAULT_REG if task == "regression" else DEFAULT_CLF)
    rows, fitted = [], {}
    for name in models:
        if name.startswith("Support Vector") and len(X_tr) > 6000:
            continue  # SVMs scale poorly
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", registry[name]())])
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        pred = pipe.predict(X_te)
        row = {"Model": name, "Train time (s)": round(fit_s, 3)}
        if task == "regression":
            row.update({
                "R²": r2_score(y_te, pred),
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "MSE": mean_squared_error(y_te, pred),
            })
        else:
            avg = "binary" if y_enc.nunique() == 2 else "weighted"
            row.update({
                "Accuracy": accuracy_score(y_te, pred),
                "Precision": precision_score(y_te, pred, average=avg, zero_division=0),
                "Recall": recall_score(y_te, pred, average=avg, zero_division=0),
                "F1": f1_score(y_te, pred, average=avg, zero_division=0),
            })
            if y_enc.nunique() == 2 and hasattr(pipe, "predict_proba"):
                try:
                    row["ROC AUC"] = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
                except ValueError:
                    pass
        if cv_folds and cv_folds > 1:
            cv = (KFold(cv_folds, shuffle=True, random_state=random_state) if task == "regression"
                  else StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
                  if stratify is not None else KFold(cv_folds, shuffle=True, random_state=random_state))
            scoring = "r2" if task == "regression" else "accuracy"
            try:
                scores = cross_val_score(Pipeline([("prep", make_preprocessor(X)), ("model", registry[name]())]),
                                         X, y_enc, cv=cv, scoring=scoring, n_jobs=None)
                row[f"CV {scoring} (mean ± sd)"] = f"{scores.mean():.3f} ± {scores.std():.3f}"
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
        fitted[name] = (pipe, pred)

    board = pd.DataFrame(rows)
    key = "R²" if task == "regression" else "F1"
    board = board.sort_values(key, ascending=False).reset_index(drop=True)
    best = board.iloc[0]["Model"]
    best_pipe, best_pred = fitted[best]
    return {
        "task": task, "target": target, "features": features, "leaderboard": board, "best_name": best,
        "best_model": best_pipe, "models": {k: v[0] for k, v in fitted.items()},
        "X_test": X_te, "y_test": y_te, "y_pred": best_pred, "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_) if label_encoder is not None else None,
        "n_train": len(X_tr), "n_test": len(X_te),
    }


def feature_importance(result, n_repeats=5, max_rows=2000) -> pd.DataFrame:
    pipe = result["best_model"]
    X, y = result["X_test"], result["y_test"]
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=0)
        y = y.loc[X.index]
    imp = permutation_importance(pipe, X, y, n_repeats=n_repeats, random_state=0, n_jobs=None)
    return pd.DataFrame({"Feature": X.columns, "Importance": imp.importances_mean, "Std": imp.importances_std}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)


def regression_figures(result, template="plotly_white"):
    y, p = np.asarray(result["y_test"]), np.asarray(result["y_pred"])
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    f1 = px.scatter(x=y, y=p, labels={"x": "Actual", "y": "Predicted"}, opacity=.7, template=template,
                    title=f"Actual vs predicted — {result['best_name']}")
    f1.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#FF5630", dash="dash"))
    resid = y - p
    f2 = px.histogram(x=resid, nbins=40, template=template, title="Residual distribution",
                      labels={"x": "Residual (actual − predicted)"}, color_discrete_sequence=["#06B6D4"])
    return f1, f2


def classification_figures(result, template="plotly_white"):
    y, p = result["y_test"], result["y_pred"]
    classes = result["classes"]
    cm = confusion_matrix(y, p)
    f1 = px.imshow(cm, text_auto=True, x=classes, y=classes, color_continuous_scale="Teal",
                   labels=dict(x="Predicted", y="Actual", color="Count"), template=template,
                   title=f"Confusion matrix — {result['best_name']}")
    f2 = None
    pipe = result["best_model"]
    if len(classes) == 2 and hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(result["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})", line=dict(color="#4F46E5", width=3)))
        f2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        f2.update_layout(template=template, title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return f1, f2


def predict_one(result, values: dict):
    row = pd.DataFrame([values])[result["features"]]
    for c in row.columns:
        ref = result["X_test"][c]
        if pd.api.types.is_numeric_dtype(ref):
            row[c] = pd.to_numeric(row[c], errors="coerce")
        else:
            row[c] = row[c].astype(str)
    pipe = result["best_model"]
    pred = pipe.predict(row)[0]
    out = {"prediction": pred}
    if result["task"] == "classification":
        out["prediction"] = result["label_encoder"].inverse_transform([int(pred)])[0]
        if hasattr(pipe, "predict_proba"):
            out["probabilities"] = dict(zip(result["classes"], pipe.predict_proba(row)[0].round(4)))
    return out


def model_to_bytes(result) -> bytes:
    payload = {k: result[k] for k in ("task", "target", "features", "best_name", "best_model", "label_encoder", "classes")}
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return buf.getvalue()


# ---------------------------------------------------------------- clustering
def _matrix(df, features):
    X = df[features].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    prep = make_preprocessor(X)
    return prep.fit_transform(X), X.index


def elbow_analysis(df, features, k_max=10):
    M, _ = _matrix(df, features)
    if len(M) > 5000:
        M = M[np.random.default_rng(0).choice(len(M), 5000, replace=False)]
    rows = []
    for k in range(2, min(k_max, len(M) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(M)
        sil = silhouette_score(M, km.labels_, sample_size=min(2000, len(M)), random_state=0)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


def cluster(df, features, algorithm="K-Means", k=3, eps=0.8, min_samples=8):
    M, idx = _matrix(df, features)
    if algorithm == "K-Means":
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(M)
    elif algorithm == "Agglomerative":
        if len(M) > 8000:
            raise ValueError("Agglomerative clustering is limited to 8,000 rows; filter the data or use K-Means.")
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(M)
    elif algorithm == "DBSCAN":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(M)
    else:
        raise ValueError(algorithm)
    sil = None
    valid = labels != -1
    if len(set(labels[valid])) > 1:
        sil = float(silhouette_score(M[valid], labels[valid], sample_size=min(3000, valid.sum()), random_state=0))
    coords = PCA(n_components=2, random_state=0).fit_transform(M) if M.shape[1] >= 2 else np.c_[M[:, 0], np.zeros(len(M))]
    out = df.loc[idx].copy()
    out["cluster"] = np.where(labels == -1, "Noise", "Cluster " + pd.Series(labels).astype(str).values)
    out["_pc1"], out["_pc2"] = coords[:, 0], coords[:, 1]
    return out, sil


def cluster_profile(clustered: pd.DataFrame, features):
    num = [c for c in features if pd.api.types.is_numeric_dtype(clustered[c])]
    prof = clustered.groupby("cluster")[num].mean().round(3) if num else pd.DataFrame(index=clustered["cluster"].unique())
    prof.insert(0, "Size", clustered["cluster"].value_counts())
    prof.insert(1, "Share %", (100 * prof["Size"] / prof["Size"].sum()).round(1))
    return prof


# ---------------------------------------------------------------- anomalies
def detect_anomalies(df, features, method="Isolation Forest", contamination=0.03, z_thresh=3.0):
    data = df[features].copy()
    if method in ("Z-score", "IQR"):
        num = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
        flags = pd.Series(False, index=data.index)
        score = pd.Series(0.0, index=data.index)
        for c in num:
            s = data[c]
            if method == "Z-score":
                z = ((s - s.mean()) / (s.std() or 1)).abs()
                flags |= z > z_thresh
                score = np.maximum(score, z.fillna(0))
            else:
                q1, q3 = s.quantile([.25, .75])
                iqr = (q3 - q1) or 1
                dist = np.maximum(q1 - s, s - q3) / iqr
                flags |= dist > 1.5
                score = np.maximum(score, dist.fillna(0))
        out = df.copy()
        out["anomaly"] = flags.fillna(False)
        out["anomaly_score"] = score.round(4)
        return out
    M, idx = _matrix(df, features)
    if method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200).fit(M)
        pred = model.predict(M)
        score = -model.score_samples(M)
    elif method == "Local Outlier Factor":
        lof = LocalOutlierFactor(n_neighbors=min(20, len(M) - 1), contamination=contamination)
        pred = lof.fit_predict(M)
        score = -lof.negative_outlier_factor_
    else:
        raise ValueError(method)
    out = df.loc[idx].copy()
    out["anomaly"] = pred == -1
    out["anomaly_score"] = np.round(score, 4)
    return out


# ---------------------------------------------------------------- PCA
def pca_analysis(df, features, n_components=None):
    M, idx = _matrix(df, features)
    n = n_components or min(10, M.shape[1])
    pca = PCA(n_components=n, random_state=0).fit(M)
    ev = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(n)],
        "Explained variance %": (100 * pca.explained_variance_ratio_).round(2),
        "Cumulative %": (100 * np.cumsum(pca.explained_variance_ratio_)).round(2),
    })
    prep = make_preprocessor(df[features].astype({c: str for c in features if not pd.api.types.is_numeric_dtype(df[c])}))
    prep.fit(df[features].astype({c: str for c in features if not pd.api.types.is_numeric_dtype(df[c])}))
    names = list(prep.get_feature_names_out())
    loadings = pd.DataFrame(pca.components_.T[:, :min(n, 5)], index=names,
                            columns=[f"PC{i + 1}" for i in range(min(n, 5))]).round(3)
    coords = pca.transform(M)
    return ev, loadings, pd.DataFrame(coords[:, :2], index=idx, columns=["PC1", "PC2"])
