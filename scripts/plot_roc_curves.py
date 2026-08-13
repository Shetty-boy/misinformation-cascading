import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ----------------- BASELINE (LR) -----------------
def get_lr_probs(split_df):
    print("Loading Baseline (LR)...")
    feature_file = "data/processed/phase06_07_features/feature_matrix.parquet"
    df = pd.read_parquet(feature_file)
    df = df[df["t_minutes"] == 120]
    merged = df.merge(split_df[["cascade_id", "split"]], on="cascade_id", how="inner")

    selected_features_path = "data/processed/phase06_07_features/selected_features.json"
    if os.path.exists(selected_features_path):
        with open(selected_features_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = ["node_count", "edge_count", "max_depth", "avg_depth", "leaf_count", "leaf_ratio",
                        "branching_factor", "root_degree", "reachable_ratio", "is_connected",
                        "tweets_per_minute", "growth_velocity", "mean_interarrival", "std_interarrival",
                        "burstiness", "cascade_age", "depth_velocity", "breadth_velocity", "branching_velocity"]

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]

    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    y_test = le.transform(test["label"])

    X_train = train[feature_cols].fillna(0).values
    X_test = test[feature_cols].fillna(0).values

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return y_test, probs

# ----------------- BiGCN -----------------
def get_bigcn_probs(unified_df, split_df):
    print("Loading BiGCN...")
    from cascade2vec.phase08_10_sota_baselines.bigcn import BiGCN, evaluate, bigcn_collate
    from adapters.bigcn_input import build_bigcn_data
    from torch_geometric.data import DataLoader as PyGDataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = "data/processed/phase08_10_sota_baselines/checkpoints/bigcn_best.pt"
    tfidf_file = "data/processed/phase08_10_sota_baselines/checkpoints/bigcn_tfidf.pkl"
    
    with open(tfidf_file, "rb") as f:
        tfidf = pickle.load(f)

    test_df = unified_df.merge(split_df[split_df["split"]=="test"][["cascade_id"]], on="cascade_id", how="inner")
    test_data = build_bigcn_data(test_df, tfidf)
    test_loader = PyGDataLoader(test_data, batch_size=64, shuffle=False, collate_fn=bigcn_collate)

    model = BiGCN(in_dim=test_data[0].x.shape[1], hidden_dim=128, num_classes=2, dropout=0.5).to(device)
    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    return np.array(metrics["labels"]), np.array(metrics["probs"])

# ----------------- Cascade2Vec -----------------
def get_c2v_probs(unified_df, split_df):
    print("Loading Cascade2Vec...")
    from cascade2vec.phase11_12_cascade2vec.cascade2vec import CASCADE2VEC, C2VClassifier, SnapshotDataset, evaluate_c2v
    from cascade2vec.phase11_12_cascade2vec.sweep import _load_best_config, _build_tfidf, SWEEP_FIXED
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = "data/processed/phase11_12_cascade2vec/checkpoints/final_model.pt"

    best_cfg = _load_best_config()
    tfidf = _build_tfidf(unified_df, split_df)

    test_ds = SnapshotDataset(unified_df, split_df, "test", tfidf, best_cfg["lam"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    encoder = CASCADE2VEC(in_dim=5000, hidden_dim=SWEEP_FIXED["hidden_dim"],
                          embed_dim=best_cfg["embed_dim"], n_layers=best_cfg["n_layers"], dropout=SWEEP_FIXED["dropout"])
    classifier = C2VClassifier(embed_dim=best_cfg["embed_dim"], num_classes=2)
    
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    
    ckpt = torch.load(ckpt_file, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])

    metrics = evaluate_c2v(encoder, classifier, test_loader, device=device, return_embeddings=True)
    return np.array(metrics["labels"]), np.array(metrics["probs"])

def main():
    unified_file = "data/processed/phase02_ingestion/unified.parquet"
    split_file = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
    
    unified_df = pd.read_parquet(unified_file)
    split_df = pd.read_parquet(split_file)

    # Dictionary to store roc data
    roc_data = {}

    try:
        y_test_lr, probs_lr = get_lr_probs(split_df)
        fpr, tpr, _ = roc_curve(y_test_lr, probs_lr)
        roc_data["Logistic Regression (Baseline)"] = (fpr, tpr, auc(fpr, tpr))
    except Exception as e:
        print(f"Failed LR: {e}")

    try:
        y_test_bigcn, probs_bigcn = get_bigcn_probs(unified_df, split_df)
        fpr, tpr, _ = roc_curve(y_test_bigcn, probs_bigcn)
        roc_data["BiGCN (SOTA)"] = (fpr, tpr, auc(fpr, tpr))
    except Exception as e:
        print(f"Failed BiGCN: {e}")

    try:
        y_test_c2v, probs_c2v = get_c2v_probs(unified_df, split_df)
        fpr, tpr, _ = roc_curve(y_test_c2v, probs_c2v)
        roc_data["Cascade2Vec"] = (fpr, tpr, auc(fpr, tpr))
    except Exception as e:
        print(f"Failed Cascade2Vec: {e}")

    if not roc_data:
        print("No models succeeded.")
        return

    # Plot
    plt.figure(figsize=(8, 6))
    
    colors = {"Logistic Regression (Baseline)": "blue", "BiGCN (SOTA)": "orange", "Cascade2Vec": "green"}
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, color=colors.get(name, "red"), lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set')
    plt.legend(loc="lower right")
    
    out_path = "logs/visualizations/roc_curves_all_models.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved ROC plot to {out_path}")

if __name__ == "__main__":
    main()
