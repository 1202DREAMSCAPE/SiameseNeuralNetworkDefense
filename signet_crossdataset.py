
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_base_network_signet_dilated
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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

def compute_accuracy_roc(predictions, labels):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    step = 0.001
    max_acc = 0
    best_threshold = dmin

    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        acc = 0.5 * (tpr + tnr)

        if acc > max_acc:
            max_acc = acc
            best_threshold = d

    return max_acc, best_threshold

cross_results = []

print("\n🧪 Searching for saved base network weights...")
saved_models = {}
for dataset_name in datasets:
    weight_file = f"{dataset_name}_base_network.weights.h5"
    if os.path.exists(weight_file):
        saved_models[dataset_name] = weight_file
        print(f"✅ Found: {weight_file}")
    else:
        print(f"⚠️ No weights found for {dataset_name}")

for train_name, weight_path in saved_models.items():
    print(f"\n🔁 Using model trained on {train_name}")
    model = create_base_network_signet_dilated((155, 220, 3), embedding_dim=EMBEDDING_SIZE)
    model.load_weights(weight_path)

    for test_name, config in datasets.items():
        print(f"🔎 Evaluating on {test_name}")
        generator = SignatureDataGenerator(
            dataset={test_name: config},
            img_height=155,
            img_width=220,
            batch_sz=32
        )

        test_imgs, test_labels = generator.get_unbatched_data()
        embeddings = model.predict(test_imgs, verbose=0)

        genuine_embs = embeddings[np.array(test_labels) == 0]
        ref = np.mean(genuine_embs, axis=0)
        distances = np.linalg.norm(embeddings - ref, axis=1)

        # True: 1 for Genuine, 0 for Forged
        labels = np.array(test_labels)
        acc, best_threshold = compute_accuracy_roc(distances, labels)
        predictions = (distances < best_threshold).astype(int)

        f1 = f1_score(labels, predictions)
        roc_auc = roc_auc_score(labels, -distances)

        print(f"✅ {train_name} → {test_name} | Acc={acc:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f} | Best Threshold={best_threshold:.4f}")
        # Save per-sample predictions
        results_per_sample = []
        for i, d in enumerate(distances):
            results_per_sample.append({
                "Train Dataset": train_name,
                "Test Dataset": test_name,
                "True Label": int(test_labels[i]),
                "Predicted Label": int(predictions[i]),
                "Distance": float(d)
            })

        # Save to CSV
        csv_name = f"{train_name}_to_{test_name}_sample_predictions.csv"
        pd.DataFrame(results_per_sample).to_csv(csv_name, index=False)
        print(f"📄 Saved detailed predictions to: {csv_name}")

        cross_results.append([train_name, test_name, acc, f1, roc_auc, best_threshold])

# Save results
if cross_results:
    df = pd.DataFrame(cross_results, columns=["Train Dataset", "Test Dataset", "Accuracy", "F1 Score", "ROC AUC", "Best Threshold"])
    df.to_csv("signet_style_cross_eval.csv", index=False)

    plt.figure(figsize=(8, 6))
    heatmap_data = df.pivot(index="Train Dataset", columns="Test Dataset", values="Accuracy")
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Greens")
    plt.title("Cross-Dataset Accuracy (SigNet ROC Threshold)")
    plt.tight_layout()
    plt.savefig("signet_style_cross_heatmap.png")
    print("✅ Evaluation CSV + heatmap saved.")
else:
    print("⚠️ No evaluation results saved.")
