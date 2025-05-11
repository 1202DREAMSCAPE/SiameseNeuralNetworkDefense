import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score, precision_score,
    confusion_matrix, roc_curve, silhouette_score
)
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from SigNet_v1 import create_base_network_signet_dilated as create_base_network_signet
from SignatureDataGenerator import SignatureDataGenerator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

IMG_SHAPE = (155, 220, 3)
BATCH_SIZE = 32
TRAINED_MODEL_WEIGHTS = "BHSig260_Bengali_base_network.weights.h5"  # ← set to your trained base weights

# Load base network
base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=128)
base_network.build(input_shape=(None, *IMG_SHAPE))
base_network.load_weights(TRAINED_MODEL_WEIGHTS)

# Datasets to evaluate on
datasets = {
    #  "CEDAR": {
    #      "path": "Dataset/CEDAR",
    #      "train_writers": list(range(260, 300)),
    #      "test_writers": list(range(300, 315))
    #  },
     "BHSig260_Bengali": {
         "path": "Dataset/BHSig260_Bengali",
         "train_writers": list(range(1, 71)),
         "test_writers": list(range(71, 101))
     },
    #   "BHSig260_Hindi": {
    #       "path": "Dataset/BHSig260_Hindi",
    #       "train_writers": list(range(101, 191)), 
    #       "test_writers": list(range(191, 260))   
    #  },
}
# === LOG TRAINING DATA DISTRIBUTION ===
def log_training_distribution(datasets, output_dir="balance_logs"):
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, config in datasets.items():
        train_writers = config.get("train_writers", [])
        if not train_writers:
            print(f"⚠️ No train writers for {dataset_name}, skipping balance check.")
            continue

        print(f"📝 Logging train writer balance for {dataset_name}...")
        generator = SignatureDataGenerator(
            dataset={dataset_name: {
                "path": config["path"],
                "train_writers": train_writers,
                "test_writers": []
            }},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE
        )

        rows = [["Writer_ID", "Genuine_Count", "Forged_Count"]]
        for dataset_path, writer_id in generator.train_writers:
            g_dir = os.path.join(dataset_path, f"writer_{writer_id:03d}", "genuine")
            f_dir = os.path.join(dataset_path, f"writer_{writer_id:03d}", "forged")

            g_count = len(os.listdir(g_dir)) if os.path.exists(g_dir) else 0
            f_count = len(os.listdir(f_dir)) if os.path.exists(f_dir) else 0
            rows.append([writer_id, g_count, f_count])

        # Save to CSV
        csv_path = os.path.join(output_dir, f"{dataset_name}_train_balance.csv")
        with open(csv_path, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"✅ Saved balance log: {csv_path}")

def compute_accuracy_roc(predictions, labels):
    dmin, dmax = np.min(predictions), np.max(predictions)
    nsame, ndiff = np.sum(labels == 1), np.sum(labels == 0)
    step = 0.00001  # or step = 1e-5
    max_acc, best_threshold = 0, 0
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions <= d
        idx2 = predictions > d
        tpr = np.sum(labels[idx1] == 1) / nsame if nsame > 0 else 0
        tnr = np.sum(labels[idx2] == 0) / ndiff if ndiff > 0 else 0
        acc = 0.5 * (tpr + tnr)
        if acc > max_acc:
            max_acc, best_threshold = acc, d
    return max_acc, best_threshold

def add_noise_to_image(img, noise_level=0.1):
    noise = np.random.normal(0, noise_level, img.shape)
    return np.clip(img + noise, 0, 1)


log_training_distribution(datasets)

for dataset_name, config in datasets.items():
    print(f"\n🔍 Evaluating on {dataset_name}...")

    # Load data
    generator = SignatureDataGenerator(
        dataset={dataset_name: {
            "path": config["path"],
            "train_writers": [],
            "test_writers": config["test_writers"]
        }},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE
    )
    X_clahe = np.load(f"balanced_data/{dataset_name}_X_bal.npy")
    y_bal = np.load(f"balanced_data/{dataset_name}_y_bal.npy")
    wids_bal = np.load(f"balanced_data/{dataset_name}_writer_ids.npy")

    embeddings = base_network.predict(X_clahe)
    test_images = X_clahe
    test_labels = y_bal


    # === SOP1: Pairwise distances
    genuine_d, forged_d, binary_labels, distances = [], [], [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            label = 1 if test_labels[i] == test_labels[j] else 0
            distances.append(d)
            binary_labels.append(label)
            (genuine_d if label == 1 else forged_d).append(d)

    _, threshold = compute_accuracy_roc(np.array(distances), np.array(binary_labels))
    preds = [1 if d <= threshold else 0 for d in distances]
    cm = confusion_matrix(binary_labels, preds)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    try:
        eer_func = interp1d(*roc_curve(binary_labels, distances)[:2])
        eer = brentq(lambda x: 1 - x - eer_func(x), 0., 1.)
    except Exception:
        eer = -1

    sop1 = {
        'SOP1_Mean_Genuine': np.mean(genuine_d),
        'SOP1_Mean_Forged': np.mean(forged_d),
        'SOP1_Std_Genuine': np.std(genuine_d),
        'SOP1_Std_Forged': np.std(forged_d),
        'SOP1_Threshold': threshold,
        'SOP1_EER': eer,
        'SOP1_AUC_ROC': roc_auc_score(binary_labels, -np.array(distances)),
        'SOP1_FAR': FP / (FP + TN) if (FP + TN) > 0 else np.nan,
        'SOP1_FRR': FN / (FN + TP) if (FN + TP) > 0 else np.nan,
        'Accuracy': accuracy_score(binary_labels, preds),
        'F1 Score': f1_score(binary_labels, preds),
        'ROC AUC': roc_auc_score(binary_labels, -np.array(distances)),
        'Recall': recall_score(binary_labels, preds),
        'Precision': precision_score(binary_labels, preds),
        'True Positives': TP,
        'True Negatives': TN,
        'False Positives': FP,
        'False Negatives': FN
    }

    # === SOP2
    sample_size = min(1000, len(embeddings))
    sop2 = {}
    if sample_size > 10:
        sample_idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
        unique_labels = np.unique(test_labels[sample_idx])
        sop2['SOP2_Silhouette'] = (
            silhouette_score(embeddings[sample_idx], test_labels[sample_idx])
            if len(unique_labels) > 1 else -1
        )
        sop2['SOP2_IntraClass_Var'] = np.mean([
            np.var(embeddings[test_labels == i], axis=0).mean()
            for i in np.unique(test_labels)
        ])

    # === SOP3
    noisy_images = np.array([add_noise_to_image(img) for img in test_images])
    noisy_embeddings = base_network.predict(noisy_images)

    ref_emb = np.mean(embeddings[test_labels == 0], axis=0)
    clean_dists = np.linalg.norm(embeddings - ref_emb, axis=1)
    noisy_dists = np.linalg.norm(noisy_embeddings - ref_emb, axis=1)

    clean_preds = (clean_dists <= threshold).astype(int)
    noisy_preds = (noisy_dists <= threshold).astype(int)

    sop3 = {
        'SOP3_Mean_Shift': np.mean(np.linalg.norm(embeddings - noisy_embeddings, axis=1)),
        'SOP3_Max_Shift': np.max(np.linalg.norm(embeddings - noisy_embeddings, axis=1)),
        'SOP3_Clean_Accuracy': accuracy_score(test_labels, clean_preds),
        'SOP3_Noisy_Accuracy': accuracy_score(test_labels, noisy_preds),
        'SOP3_Accuracy_Drop': accuracy_score(test_labels, clean_preds) - accuracy_score(test_labels, noisy_preds),
        'SOP3_Threshold_Used': threshold
    }

    # === Save to TXT
    output_file = f"evaluation_{dataset_name}.txt"
    with open(output_file, "w") as f:
        f.write("=== SOP1 ===\n")
        for k, v in sop1.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== SOP2 ===\n")
        for k, v in sop2.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== SOP3 ===\n")
        for k, v in sop3.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Saved: {output_file}")

    # ========== Visualization ==========
    plt.figure(figsize=(15, 5))

    # --- Distance Distribution ---
    plt.subplot(131)
    plt.hist(genuine_d, bins=30, alpha=0.6, label='Genuine')
    plt.hist(forged_d, bins=30, alpha=0.6, label='Forged')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.3f}')
    plt.title(f"{dataset_name} Distance Distribution")
    plt.legend()

    # --- Embedding Space (PCA) ---
    plt.subplot(132)
    pca = PCA(n_components=2).fit_transform(embeddings[:200])
    plt.scatter(pca[:, 0], pca[:, 1], c=test_labels[:200], cmap='coolwarm', alpha=0.6)
    plt.title("Embedding Space (PCA)")

    # --- Embedding Shifts from Noise ---
    if 'SOP3_Mean_Shift' in sop3 and sop3['SOP3_Mean_Shift'] != -1:
        plt.subplot(133)
        shifts = np.linalg.norm(embeddings - noisy_embeddings, axis=1)
        plt.hist(shifts, bins=20)
        plt.title("Noise-Induced Embedding Shifts")

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_baseline_metrics.png")
    plt.close()
