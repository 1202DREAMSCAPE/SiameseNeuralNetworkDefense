import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, silhouette_score, precision_score, recall_score
)
from utils import (
    visualize_pairs,
    plot_distance_distributions
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import seaborn as sns
from SignatureDataGenerator import SignatureDataGenerator
import numpy as np
from SigNet_v1 import get_triplet_loss, create_base_network_signet, create_triplet_network_from_existing_base

np.random.seed(1337)
random.seed(1337)

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), tf.keras.backend.epsilon()))

def compute_accuracy_roc(predictions, labels):
    """
    Compute ROC-based accuracy with the best threshold.
    Returns:
        max_acc: Best accuracy found
        best_threshold: Distance threshold yielding best ROC-based accuracy
    """
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
   
    step = 0.001
    max_acc = 0
    best_threshold = 0.0

    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d
       
        tpr = float(np.sum(labels[idx1] == 1)) / nsame if nsame > 0 else 0
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff if ndiff > 0 else 0
        acc = 0.5 * (tpr + tnr)
       
        if acc > max_acc:
            max_acc = acc
            best_threshold = d

    return max_acc, best_threshold

def plot_far_frr_bar_chart(roc_far, roc_frr, dataset_name='Dataset', save_path=None):
    """
    Plots a bar chart comparing FAR and FRR for the ROC-based threshold.

    Parameters:
        roc_far (float): False Acceptance Rate
        roc_frr (float): False Rejection Rate
        dataset_name (str): Dataset name for the plot title
        save_path (str): Optional path to save the figure as a PNG
    """
    labels = ['FAR', 'FRR']
    values = [roc_far, roc_frr]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=['skyblue', 'salmon'])

    ax.set_ylim(0, 1)
    ax.set_ylabel('Rate')
    ax.set_title(f'{dataset_name} - FAR vs FRR (ROC Threshold)')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def evaluate_sop1(genuine_d, forged_d, binary_labels, distances, threshold):
    metrics = {
        'SOP1_Mean_Genuine': np.mean(genuine_d),
        'SOP1_Mean_Forged': np.mean(forged_d),
        'SOP1_Std_Genuine': np.std(genuine_d),
        'SOP1_Std_Forged': np.std(forged_d),
        'SOP1_Threshold': threshold
    }

    # ROC AUC and EER
    try:
        fpr, tpr, thresholds = roc_curve(binary_labels, distances)
        fnr = 1 - tpr
        eer_threshold_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_threshold_idx]

        metrics['SOP1_EER'] = eer
        metrics['SOP1_AUC_ROC'] = roc_auc_score(binary_labels, -np.array(distances))  # Inverted for "genuine < forged"
    except Exception:
        metrics['SOP1_EER'] = -1
        metrics['SOP1_AUC_ROC'] = -1

    # Apply externally computed threshold
    preds = [1 if d <= threshold else 0 for d in distances]
    cm = confusion_matrix(binary_labels, preds)

    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        metrics['SOP1_FAR'] = FP / (FP + TN) if (FP + TN) > 0 else np.nan
        metrics['SOP1_FRR'] = FN / (FN + TP) if (FN + TP) > 0 else np.nan
    else:
        metrics['SOP1_FAR'], metrics['SOP1_FRR'] = np.nan, np.nan

    return metrics

def add_extended_sop1_metrics(sop1_metrics, binary_labels, distances, threshold):
    """
    Adds TPR, TNR, Precision, and Recall to the SOP1 metrics dictionary.

    Parameters:
        sop1_metrics (dict): Existing SOP1 metrics dictionary.
        binary_labels (list or np.array): True binary labels (1 = genuine, 0 = forged).
        distances (list or np.array): Computed distances between embedding pairs.
        threshold (float): Distance threshold for classification.

    Returns:
        dict: Updated SOP1 metrics with additional diagnostic metrics.
    """
    preds = (np.array(distances) <= threshold).astype(int)
    labels = np.array(binary_labels)

    TP = np.sum((labels == 1) & (preds == 1))
    TN = np.sum((labels == 0) & (preds == 0))
    FP = np.sum((labels == 0) & (preds == 1))
    FN = np.sum((labels == 1) & (preds == 0))

    sop1_metrics['SOP1_TPR'] = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # True Positive Rate
    sop1_metrics['SOP1_TNR'] = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # True Negative Rate
    sop1_metrics['SOP1_Precision'] = precision_score(labels, preds) if (TP + FP) > 0 else np.nan
    sop1_metrics['SOP1_Recall'] = recall_score(labels, preds) if (TP + FN) > 0 else np.nan

    return sop1_metrics

def evaluate_sop2(embeddings, labels):
    metrics = {}
    sample_size = min(1000, len(embeddings))
    if sample_size > 10:
        sample_idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
        if len(np.unique(labels[sample_idx])) > 1:
            metrics['SOP2_Silhouette'] = silhouette_score(embeddings[sample_idx], labels[sample_idx])
        else:
            metrics['SOP2_Silhouette'] = -1
        metrics['SOP2_IntraClass_Var'] = np.mean([np.var(embeddings[labels == i], axis=0).mean()
                                                  for i in np.unique(labels)])
    return metrics

def calculate_psnr(clean_img, noisy_img):
    mse = np.mean((clean_img * 255 - noisy_img * 255) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

def add_noise_to_image(img, noise_level=0.1):
    """Add Gaussian noise to a single image"""
    noise = np.random.normal(0, noise_level, img.shape)
    return np.clip(img + noise, 0, 1)

def evaluate_sop3(clean_emb, noisy_emb, clean_labels, threshold):
    """
    Corrected SOP3 evaluation with:
    - Proper accuracy drop calculation
    - Valid PSNR computation
    - Robust shift metrics
    """
    # Check if clean_labels are binary
    assert set(np.unique(clean_labels)) == {0, 1}, "Labels must be binary 0=forged, 1=genuine"

    # 1. Calculate embedding shifts
    shifts = np.linalg.norm(clean_emb - noisy_emb, axis=1)
    
    # 2. Compute verification decisions using reference signature
    ref = np.mean(clean_emb[clean_labels == 1], axis=0)
    
    # Clean and noisy distances to reference
    clean_dists = np.linalg.norm(clean_emb - ref, axis=1)
    noisy_dists = np.linalg.norm(noisy_emb - ref, axis=1)
    
    # Predictions (1=genuine, 0=forged)
    clean_preds = (clean_dists <= threshold).astype(int)
    noisy_preds = (noisy_dists <= threshold).astype(int)
    
    # 3. Calculate actual accuracy drop (should be negative)
    clean_acc = accuracy_score(clean_labels, clean_preds)
    noisy_acc = accuracy_score(clean_labels, noisy_preds)
    acc_drop = clean_acc - noisy_acc
    
    return {
        'SOP3_Mean_Shift': np.mean(shifts),
        'SOP3_Max_Shift': np.max(shifts),
        'SOP3_Clean_Accuracy': clean_acc,
        'SOP3_Noisy_Accuracy': noisy_acc,
        'SOP3_Accuracy_Drop': acc_drop,
        'SOP3_Threshold_Used': threshold
    }


# ========== CONFIG ==========
IMG_SHAPE = (155, 220, 3)
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 1.0

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
     }
}

results = []

# === Triplet Loss Training with Hard Negative Mining (Baseline Replacement) ===
for dataset_name, config in datasets.items():
    print(f"\n📦 Training Triplet Model for {dataset_name}")

    try:
        # Setup generator
        generator = SignatureDataGenerator(
            dataset={dataset_name: config},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE
        )

        # Create base network (SigNet)
        base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=128)

        # Generate triplets using hard negative mining
        anchor_list, positive_list, negative_list = generator.generate_hard_mined_triplets(
            base_network=base_network,
            batch_size=BATCH_SIZE
        )

        if len(anchor_list) == 0:
            print("⚠ No triplets generated. Skipping.")
            continue

        # Convert to tensors
        anc = np.array(anchor_list)
        pos = np.array(positive_list)
        neg = np.array(negative_list)

        train_data = tf.data.Dataset.from_tensor_slices(
            ({"input_anchor": anc, "input_positive": pos, "input_negative": neg}, np.zeros(len(anc)))
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Build Triplet Model
        triplet_model = create_triplet_network_from_existing_base(base_network)
        triplet_model.compile(optimizer=RMSprop(0.001), loss=get_triplet_loss(margin=MARGIN))

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)

        # Train the model
        print("🚀 Training started...")
        start_time = time.time()
        history = triplet_model.fit(train_data, epochs=EPOCHS, callbacks=[early_stop])
        train_time = time.time() - start_time

        # Save weights and model
        weights_dir = "base_weights"
        os.makedirs(weights_dir, exist_ok=True)
        base_network.save_weights(f"{weights_dir}/{dataset_name}_signet_triplet.weights.h5")
        triplet_model.save(f"{dataset_name}_triplet_model.h5")

        print(f"✅ Saved model and weights for {dataset_name}")

    except Exception as e:
        print(f"❌ Training failed for {dataset_name}")
        print(e)
        continue  # Skip evaluation if training failed

    # === Evaluation ===
    print(f"\n🔍 Starting Evaluation for {dataset_name}")

    # 1. Load test and noisy data
    test_images, test_labels = generator.get_unbatched_data()
    noisy_imgs, _ = generator.get_unbatched_data(noisy=True)

    # 2. Generate embeddings
    embeddings = base_network.predict(test_images, verbose=0)
    noisy_emb = base_network.predict(noisy_imgs, verbose=0)

    # ========== SOP 1 Evaluation ==========
    genuine_d, forged_d, distances, binary_labels = [], [], [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            label_i = test_labels[i]
            label_j = test_labels[j]

            if label_i == label_j:  # same writer → genuine
                genuine_d.append(dist)
            else:  # different writers → forged
                forged_d.append(dist)

    # Combine distances and labels
    distances = np.array(genuine_d + forged_d)
    binary_labels = np.array([1] * len(genuine_d) + [0] * len(forged_d))

    # Compute ROC-based accuracy and best threshold
    acc, best_threshold = compute_accuracy_roc(distances, binary_labels)
    preds = (distances <= best_threshold).astype(int)
    f1 = f1_score(binary_labels, preds) 

    #print(f"✅ Threshold: {best_threshold:.4f} | F1: {best_f1:.4f} | Accuracy: {acc:.4f}")

    # 5. Save embeddings and threshold
    ref_dir = "ref_labels_embeds"
    os.makedirs(ref_dir, exist_ok=True)
    np.save(os.path.join(ref_dir, f"{dataset_name}_ref_embs.npy"), embeddings)
    np.save(os.path.join(ref_dir, f"{dataset_name}_ref_labels.npy"), test_labels)
    with open(os.path.join(ref_dir, f"{dataset_name}_threshold.txt"), "w") as f:
        f.write(str(best_threshold))

    # 6. SOP 1: Metric calculations
    sop1_metrics = evaluate_sop1(genuine_d, forged_d, binary_labels, distances, best_threshold)
    sop1_metrics = add_extended_sop1_metrics(sop1_metrics, binary_labels, distances, best_threshold)
    plot_far_frr_bar_chart(sop1_metrics['SOP1_FAR'], sop1_metrics['SOP1_FRR'], dataset_name, f"{dataset_name}_ROC_FAR_FRR_BarChart_Triplet.png")

    # 7. SOP 2: Embedding clustering
    sop2_metrics = evaluate_sop2(embeddings, test_labels)

    # 8. SOP 3: Noise robustness
    try:
        sop3_metrics = evaluate_sop3(clean_emb=embeddings, noisy_emb=noisy_emb, clean_labels=test_labels, threshold=best_threshold)
        psnr_values = [calculate_psnr(c, n) for c, n in zip(test_images, noisy_imgs)]
        sop3_metrics['SOP3_Mean_PSNR'] = np.mean(psnr_values)
    except Exception as e:
        print(f"⚠ SOP3 failed: {e}")
        sop3_metrics = {k: -1 for k in ['SOP3_Mean_PSNR', 'SOP3_Accuracy_Drop', 'SOP3_Mean_Shift', 'SOP3_Max_Shift']}

    # 9. Save metrics
    results.append({
        "Dataset": dataset_name,
        "Training_Time": train_time,
        **sop1_metrics,
        **sop2_metrics,
        **sop3_metrics,
        "Accuracy": acc,
        "F1_Score": f1
    })
    pd.DataFrame(results).to_csv("SigNet_Triplet_SOP_Results.csv", index=False)
    print(f"📊 Metrics saved for {dataset_name}")

    # 10. Visualizations
    plt.figure(figsize=(15, 5))

    # Distance distributions
    plt.subplot(131)
    plt.hist(genuine_d, bins=30, alpha=0.6, label='Genuine')
    plt.hist(forged_d, bins=30, alpha=0.6, label='Forged')
    plt.axvline(sop1_metrics['SOP1_Threshold'], color='r', linestyle='--')
    plt.title(f"{dataset_name} Distance Distribution")
    plt.legend()

    # PCA projection
    plt.subplot(132)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit_transform(embeddings[:200])
    plt.scatter(pca[:, 0], pca[:, 1], c=test_labels[:200], cmap='coolwarm', alpha=0.6)
    plt.title("Embedding Space (PCA)")

    # Embedding shift
    if 'SOP3_Mean_Shift' in sop3_metrics and sop3_metrics['SOP3_Mean_Shift'] != -1:
        plt.subplot(133)
        shifts = np.linalg.norm(embeddings - noisy_emb, axis=1)
        plt.hist(shifts, bins=20)
        plt.title("Noise-Induced Embedding Shifts")

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_triplet_metrics.png")
    plt.close()

    print(f"📋 Finished evaluation for Triplet {dataset_name}. CSV and plots updated.\n")
