# evaluate_raw_with_smote.py
# Evaluates performance on raw data after applying SMOTE to embeddings

import os
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from SigNet_v1 import create_base_network_signet_dilated as create_base_network_signet
from SignatureDataGenerator import SignatureDataGenerator

EMBEDDING_SIZE = 128

datasets = {
    "CEDAR": {
        "path": "Dataset/CEDAR",
        "train_writers": list(range(260, 300)),
        "test_writers": list(range(300, 315))
    },
    "BHSig260_Bengali": {
        "path": "Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),
        "test_writers": list(range(71, 101))
    },
    "BHSig260_Hindi": {
        "path": "Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 191)),
        "test_writers": list(range(191, 260))
    },
}

results = []

for dataset_name, config in datasets.items():
    print(f"\n[INFO] Processing {dataset_name} (Raw + SMOTE)")

    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=155,
        img_width=220,
        batch_sz=32
    )

    base_model = create_base_network_signet((155, 220, 3), embedding_dim=EMBEDDING_SIZE)
    base_model.load_weights(f"{dataset_name}.weights.h5")

    images, labels = generator.get_all_data_with_labels()
    embeddings = base_model.predict(images)

    label_counts = Counter(labels)
    print(f"[INFO] Original distribution: {label_counts}")

    # Split original embeddings
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    clf_orig = LogisticRegression(max_iter=1000)
    clf_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = clf_orig.predict(X_test_orig)
    y_proba_orig = clf_orig.predict_proba(X_test_orig)[:, 1]

    # Save original model
    joblib.dump(clf_orig, f"logreg_{dataset_name}_original.pkl")

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    smote_embeddings, smote_labels = smote.fit_resample(embeddings, labels)
    print(f"[INFO] After SMOTE: {Counter(smote_labels)}")

    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(smote_embeddings, smote_labels, test_size=0.2, random_state=42)
    clf_smote = LogisticRegression(max_iter=1000)
    clf_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = clf_smote.predict(X_test_smote)
    y_proba_smote = clf_smote.predict_proba(X_test_smote)[:, 1]

    # Save SMOTE model
    joblib.dump(clf_smote, f"logreg_{dataset_name}_smote.pkl")

    # Metrics
    def get_metrics(y_true, y_pred, y_proba):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_proba)
        }

    metrics_orig = get_metrics(y_test_orig, y_pred_orig, y_proba_orig)
    metrics_smote = get_metrics(y_test_smote, y_pred_smote, y_proba_smote)

    print(f"\nOriginal Metrics: {metrics_orig}")
    print(f"SMOTE Metrics: {metrics_smote}")

    # Add to results
    results.append([dataset_name] + list(metrics_orig.values()) + list(metrics_smote.values()))

    # Precision-Recall Curve
    precision_orig, recall_orig, _ = precision_recall_curve(y_test_orig, y_proba_orig)
    precision_smote, recall_smote, _ = precision_recall_curve(y_test_smote, y_proba_smote)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_orig, precision_orig, label="Original")
    plt.plot(recall_smote, precision_smote, label="SMOTE")
    plt.title(f"Precision-Recall Curve: {dataset_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_precision_recall_curve.png")
    plt.close()

    # Confusion Matrix
    cm_orig = confusion_matrix(y_test_orig, y_pred_orig)
    cm_smote = confusion_matrix(y_test_smote, y_pred_smote)
    print(f"Confusion Matrix (Original):\n{cm_orig}")
    print(f"Confusion Matrix (SMOTE):\n{cm_smote}")

# Save results
columns = ["Dataset"] + [f"Original_{m}" for m in metrics_orig.keys()] + [f"SMOTE_{m}" for m in metrics_smote.keys()]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("smote_comparison_metrics.csv", index=False)
print("\n[INFO] SMOTE vs Original metrics saved to smote_comparison_metrics.csv")