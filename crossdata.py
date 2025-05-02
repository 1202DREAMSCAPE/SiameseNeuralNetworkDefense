
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_base_network_signet_dilated
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Updated fixed threshold
FIXED_THRESHOLD = 0.30
EMBEDDING_SIZE = 128

# Dataset configurations
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

print("\n🧪 Searching for saved base network weights...")
saved_models = {}

for dataset_name in datasets:
    weight_file = f"{dataset_name}_base_network.weights.h5"
    if os.path.exists(weight_file):
        saved_models[dataset_name] = weight_file
        print(f"✅ Found: {weight_file}")
    else:
        print(f"⚠️ No weights found for {dataset_name}")

cross_results = []

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

        predictions = (distances < FIXED_THRESHOLD).astype(int)
        acc = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, -distances)
        cm = confusion_matrix(test_labels, predictions)

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

        # Save to CSV per test dataset
        csv_name = f"{train_name}_to_{test_name}_predictions.csv"
        pd.DataFrame(results_per_sample).to_csv(csv_name, index=False)
        print(f"📄 Saved detailed predictions: {csv_name}")

        print("Confusion Matrix:\n", cm)
        print(f"✅ {train_name} → {test_name} | Acc={acc:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}")
        cross_results.append([train_name, test_name, acc, f1, roc_auc, FIXED_THRESHOLD])

# Save results
if cross_results:
    df = pd.DataFrame(cross_results, columns=["Train Dataset", "Test Dataset", "Accuracy", "F1 Score", "ROC AUC", "Threshold Used"])
    df.to_csv("cross_dataset_eval_fixed.csv", index=False)

    plt.figure(figsize=(8, 6))
    heatmap_data = df.pivot(index="Train Dataset", columns="Test Dataset", values="Accuracy")
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Blues")
    plt.title("Cross-Dataset Accuracy")
    plt.tight_layout()
    plt.savefig("cross_dataset_fixed_threshold_heatmap.png")
    print("✅ Evaluation CSV + heatmap saved.")
else:
    print("⚠️ No evaluation results saved.")
