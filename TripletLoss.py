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
from Signet_v1 import get_triplet_loss, create_base_network_signet

np.random.seed(1337)
random.seed(1337)

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), tf.keras.backend.epsilon()))

def compute_threshold_f1(distances, labels, num_thresholds=1000):
    """
    Sweep over 'num_thresholds' equally spaced distances between
    min(distances) and max(distances), returning the threshold
    that yields the highest F1-Score.
    """
    d = np.array(distances)
    lab = np.array(labels)

    dmin, dmax = d.min(), d.max()
    best_thr = dmin
    best_f1  = 0.0

    for thr in np.linspace(dmin, dmax, num_thresholds):
        preds = (d <= thr).astype(int)  # 1=genuine, 0=forged
        f1 = f1_score(lab, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1

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
        eer_func = interp1d(fpr, tpr)
        eer = brentq(lambda x: 1. - x - eer_func(x), 0., 1.)
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
    # 1. Calculate embedding shifts
    shifts = np.linalg.norm(clean_emb - noisy_emb, axis=1)
    
    # 2. Compute verification decisions using reference signature
    ref = np.mean(clean_emb[clean_labels == 0], axis=0)  # Mean genuine embedding
    
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
MARGIN = 0.7

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

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Triplet Model for Dataset {dataset_name}")

    # Load data and model
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    base_network = create_base_network_signet(IMG_SHAPE)

    # Generate triplets (hard-mined)
    anchor_imgs, positive_imgs, negative_imgs = generator.generate_hard_mined_triplets(base_network, batch_size=BATCH_SIZE)

    # Split into train and val
    val_split = int(0.9 * len(anchor_imgs))
    train_data = ([np.array(anchor_imgs[:val_split]), np.array(positive_imgs[:val_split]), np.array(negative_imgs[:val_split])],
                  np.ones((val_split, 1)))
    val_data = ([np.array(anchor_imgs[val_split:]), np.array(positive_imgs[val_split:]), np.array(negative_imgs[val_split:])],
                np.ones((len(anchor_imgs) - val_split, 1)))

    # Build Triplet Network
    input_a = Input(shape=IMG_SHAPE)
    input_p = Input(shape=IMG_SHAPE)
    input_n = Input(shape=IMG_SHAPE)

    emb_a = base_network(input_a)
    emb_p = base_network(input_p)
    emb_n = base_network(input_n)

    merged = layers.Lambda(lambda x: tf.stack(x, axis=1))([emb_a, emb_p, emb_n])
    model = Model(inputs=[input_a, input_p, input_n], outputs=merged)
    model.compile(optimizer=RMSprop(0.0001), loss=get_triplet_loss(MARGIN))

    # Train
    start_time = time.time()
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    train_time = time.time() - start_time

    # Save
    os.makedirs("base_weights", exist_ok=True)
    model.save_weights(f"base_weights/{dataset_name}_triplet_model.weights.h5")
    base_network.save_weights(f"base_weights/{dataset_name}_signet_triplet.weights.h5")
    model.save(f"{dataset_name}_triplet_model.h5")
    print(f"✅ Saved model and base network for {dataset_name}")

    # Evaluation
    test_images, test_labels = generator.get_unbatched_data()
    noisy_imgs, _ = generator.get_unbatched_data(noisy=True)

    embeddings = base_network.predict(test_images, verbose=0)
    noisy_emb = base_network.predict(noisy_imgs, verbose=0)

    # SOP 1
    genuine_d, forged_d = [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            label_i, label_j = test_labels[i], test_labels[j]
            if label_i == 0 and label_j == 0:
                genuine_d.append(dist)
            elif (label_i == 0 and label_j == 1) or (label_i == 1 and label_j == 0):
                forged_d.append(dist)

    distances = genuine_d + forged_d
    binary_labels = [1] * len(genuine_d) + [0] * len(forged_d)

    best_threshold, best_f1 = compute_threshold_f1(np.array(distances), np.array(binary_labels))
    preds_f1 = (np.array(distances) <= best_threshold).astype(int)
    acc = accuracy_score(binary_labels, preds_f1)

    print(f"✅ F1 Threshold: {best_threshold:.4f} | F1: {best_f1:.4f} | Accuracy: {acc:.4f}")
    # Save embeddings and labels for system inference (e.g., FastAPI backend)
    ref_dir = "ref_labels_embeds"
    os.makedirs(ref_dir, exist_ok=True)

    np.save(os.path.join(ref_dir, f"{dataset_name}_ref_embs.npy"), embeddings)
    np.save(os.path.join(ref_dir, f"{dataset_name}_ref_labels.npy"), test_labels)
    with open(os.path.join(ref_dir, f"{dataset_name}_threshold.txt"), "w") as f:
        f.write(str(best_threshold))

    sop1_metrics = evaluate_sop1(genuine_d, forged_d, binary_labels, distances, best_threshold)
    sop1_metrics = add_extended_sop1_metrics(sop1_metrics, binary_labels, distances, best_threshold)
    plot_far_frr_bar_chart(sop1_metrics['SOP1_FAR'], sop1_metrics['SOP1_FRR'], dataset_name, f"{dataset_name}_ROC_FAR_FRR_BarChart_f1.png")

    # SOP 2
    sop2_metrics = evaluate_sop2(embeddings, test_labels)

    # SOP 3
    try:
        sop3_metrics = evaluate_sop3(clean_emb=embeddings, noisy_emb=noisy_emb, clean_labels=test_labels, threshold=best_threshold)
        sop3_metrics['SOP3_Mean_PSNR'] = np.mean([calculate_psnr(c, n) for c, n in zip(test_images, noisy_imgs)])
    except Exception as e:
        print(f"⚠ SOP3 failed: {e}")
        sop3_metrics = {k: -1 for k in ['SOP3_Mean_PSNR', 'SOP3_Accuracy_Drop', 'SOP3_Mean_Shift', 'SOP3_Max_Shift']}

    # Save results
    results.append({
        "Dataset": dataset_name,
        "Training_Time": train_time,
        **sop1_metrics,
        **sop2_metrics,
        **sop3_metrics,
        "Accuracy": acc,
        "F1_Score": best_f1
    })
    pd.DataFrame(results).to_csv("SigNet_Triplet_SOP_Results.csv", index=False)
    print(f"✅ Metrics saved for {dataset_name}")

    # ========== Visualization ==========
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.hist(genuine_d, bins=30, alpha=0.6, label='Genuine')
    plt.hist(forged_d, bins=30, alpha=0.6, label='Forged')
    plt.axvline(sop1_metrics['SOP1_Threshold'], color='r', linestyle='--')
    plt.title(f"{dataset_name} Triplet Distance Distribution")
    plt.legend()

    plt.subplot(132)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit_transform(embeddings[:200])
    plt.scatter(pca[:, 0], pca[:, 1], c=test_labels[:200], alpha=0.6)
    plt.title("Embedding Space (PCA)")

    if 'SOP3_Mean_Shift' in sop3_metrics and sop3_metrics['SOP3_Mean_Shift'] != -1:
        plt.subplot(133)
        shifts = np.linalg.norm(embeddings - noisy_emb, axis=1)
        plt.hist(shifts, bins=20)
        plt.title("Noise-Induced Embedding Shifts")

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_triplet_metrics.png")
    plt.close()


print(f"📋 Finished evaluation for Triplet {dataset_name}. Current CSV updated.\n")
