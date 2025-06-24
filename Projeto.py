from pathlib import Path
import urllib.request
import pandas as pd

import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    silhouette_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import dendrogram

NSL_TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
)
NSL_TEST_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
)
NSL_COLUMNS_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/Field%20Names.txt"
)

def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✓ {dest.name} already downloaded")
        return
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)

def fetch_nsl_kdd(data_dir: Path = Path("data")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download NSL‑KDD train/test splits and return them as DataFrames."""
    fallback_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate"
    ]
    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"
    names_path = data_dir / "KDD_names.txt"

    _download(NSL_TRAIN_URL, train_path)
    _download(NSL_TEST_URL, test_path)
    try:
        _download(NSL_COLUMNS_URL, names_path)
        with names_path.open() as f:
            names = [line.split(":")[0] for line in f if not line.startswith("|")]
    except Exception as e:
        print("[warn] Could not download column names:", e)
        names = fallback_names.copy()

    # The dataset has an extra target and difficulty columns
    names += ["target", "difficulty"]

    # space‑separated ⇒ use read_csv with no header
    train = pd.read_csv(train_path, names=names)
    test = pd.read_csv(test_path, names=names)
    return train, test


# ---------------------------------------------------------------------------
# Pre‑processing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """One‑hot encode categoricals, standardise numerics, return X, y."""
    df = df.copy()
    y = df["target"].apply(lambda x: 0 if x == "normal" else 1).to_numpy()
    df.drop(columns=["target", "difficulty"], inplace=True)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
         ("num", StandardScaler(), num_cols)],
        remainder="drop",
    )
    X = ct.fit_transform(df)
    return X.astype(np.float32), y

# ---------------------------------------------------------------------------
# PCA retaining ≥ 85 % variance
# ---------------------------------------------------------------------------
def pca_fit_transform(X, var_threshold=0.85, seed=42):
    pca_full = PCA(svd_solver="full", random_state=seed).fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, var_threshold)) + 1
    pca = PCA(n_components=n_comp, random_state=seed)
    T = pca.fit_transform(X)
    return T, pca

# ---------------------------------------------------------------------------
# K‑Means scratch
# ---------------------------------------------------------------------------
def kmeans_scratch(X, K, max_iter=300, tol=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, size=X.shape[0])
    for _ in range(max_iter):
        centroids = np.vstack([X[labels == k].mean(axis=0) for k in range(K)])
        # handle empty cluster
        for k in range(K):
            if np.isnan(centroids[k]).any():
                centroids[k] = X[rng.choice(len(X))]
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.all(labels == new_labels) or np.mean(labels != new_labels) < tol:
            break
        labels = new_labels
    return labels, centroids

# ---------------------------------------------------------------------------
# Complete‑link hierarchy scratch (small N)
# ---------------------------------------------------------------------------
def linkage_complete(X):
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X, metric="euclidean"))
    np.fill_diagonal(D, np.inf)
    merges = []
    clusters = {i: [i] for i in range(len(X))}
    for t in range(len(X) - 1):
        i, j = divmod(D.argmin(), D.shape[0])
        dmin = D[i, j]
        merges.append([i, j, dmin, len(clusters[i]) + len(clusters[j])])
        new = len(D)
        clusters[new] = clusters[i] + clusters[j]
        # expand matrix
        D = np.pad(D, ((0,1),(0,1)), constant_values=np.inf)
        for k in range(new):
            if k in (i, j): continue
            dist = max(D[i, k], D[j, k])
            D[new, k] = D[k, new] = dist
        D[i, :], D[:, i] = np.inf, np.inf
        D[j, :], D[:, j] = np.inf, np.inf
    return np.array(merges)

# ---------------------------------------------------------------------------
# Isolation Forest helper
# ---------------------------------------------------------------------------
def run_isolation_forest(T, y, contamination=0.01, seed=42):
    iso = IsolationForest(
        n_estimators=100, max_samples=256,
        contamination=contamination, random_state=seed
    )
    iso.fit(T)
    scores = -iso.decision_function(T)
    preds = (iso.predict(T) == -1).astype(int)
    metrics = {}
    if y is not None and len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(y, scores)
        metrics["pr_auc"] = average_precision_score(y, scores)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        metrics["fpr"] = fp / (fp + tn + 1e-12)
    return scores, preds, metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nsl_kdd"], default="nsl_kdd")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--contamination", type=float, default=0.01)
    ap.add_argument("--var", type=float, default=0.85)
    args = ap.parse_args()

    if args.dataset == "nsl_kdd":
        train, test = fetch_nsl_kdd()
        df = pd.concat([train, test], ignore_index=True)
    else:
        raise ValueError("Unsupported dataset")

    X, y = preprocess(df)
    T, pca = pca_fit_transform(X, var_threshold=args.var)

    km_labels, _ = kmeans_scratch(T, args.k)
    print("Silhouette:", silhouette_score(T, km_labels))

    merges = linkage_complete(T[:1000]) if T.shape[0] <= 1000 else None

    scores, preds, metrics = run_isolation_forest(
        T, y, contamination=args.contamination
    )
    print("IF metrics:", metrics)

    out = Path("output"); out.mkdir(exist_ok=True)
    # scree
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("PCA Scree")
    plt.savefig(out / "pca_scree.png", dpi=150)

    if T.shape[1] >= 2:
        plt.figure()
        plt.scatter(T[:,0], T[:,1], c=km_labels, s=8, cmap="tab10")
        plt.title("PC1 × PC2 by K‑Means")
        plt.savefig(out / "kmeans_scatter.png", dpi=150)

    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("IF scores")
    plt.savefig(out / "iso_scores_hist.png", dpi=150)

    if metrics:
        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure(); plt.plot(fpr, tpr); plt.title("ROC"); plt.savefig(out / "roc.png", dpi=150)
        prec, rec, _ = precision_recall_curve(y, scores)
        plt.figure(); plt.plot(rec, prec); plt.title("PR"); plt.savefig(out / "pr.png", dpi=150)

    if merges is not None:
        plt.figure(figsize=(10,5))
        dendrogram(merges, truncate_mode="level", p=10, no_labels=True)
        plt.title("Hierarchical Dendrogram")
        plt.savefig(out / "dendrogram.png", dpi=150)

    print("Artefacts saved in ./output")

if __name__ == "__main__":
    main()
