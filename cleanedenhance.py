import os
import numpy as np
import traceback
from sklearn.decomposition import PCA
import tensorflow as tf
import time
import csv
import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score, 
    top_k_accuracy_score, silhouette_score, roc_curve
)
from utils import (
    find_hard_negatives,
    compute_hard_negative_metrics,
    evaluate_threshold_with_given_threshold,
    visualize_triplets
)
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from SigNet_v1 import get_triplet_loss
from SigNet_v1 import create_base_network_signet
from SigNet_v1 import create_triplet_network_from_existing_base
from collections import Counter
from SignatureDataGenerator import SignatureDataGenerator
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)
tf.config.set_soft_device_placement(True)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# --- Define helper ---
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

    #sweeping across num_thresholds linearly spaced distance values
    for thr in np.linspace(dmin, dmax, num_thresholds):
        #generating binary predictions based on whether each distance falls below the threshold
        preds = (d <= thr).astype(int)  # 1=genuine, 0=forged
       #computing the F1-score at each threshold
        f1 = f1_score(lab, preds)
        #if the F1-score is better than the best one so far, update the best threshold and score
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1

def compute_far_frr(y_true, y_pred):
    """
    Computes False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    """
    assert len(y_true) == len(y_pred)
    fp = np.sum((y_true == 0) & (y_pred == 1))  # Forged accepted (False Acceptance)
    fn = np.sum((y_true == 1) & (y_pred == 0))  # Genuine rejected (False Rejection)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return far, frr

def apply_partial_clahe_per_writer(generator, images, labels, writer_ids, save_dir="clahe_samples"):
    os.makedirs(save_dir, exist_ok=True)
    processed = []
    sample_log = []

    for wid in np.unique(writer_ids):
        mask = (writer_ids == wid)
        imgs = images[mask]
        lbls = labels[mask]

        genuine = imgs[lbls == 1]
        forged = imgs[lbls == 0]

        gen_idx = np.random.choice(len(genuine), size=len(genuine)//2, replace=False) if len(genuine) > 1 else []
        forg_idx = np.random.choice(len(forged), size=len(forged)//2, replace=False) if len(forged) > 1 else []

        gen_processed = []
        for i in range(len(genuine)):
            raw_img = genuine[i]
            if i in gen_idx:
                clahe_img = generator.preprocess_image_from_array(raw_img)
                gen_processed.append(clahe_img)

                # Save one sample
                if i < 1:
                    path = save_clahe_comparison(raw_img, clahe_img, save_dir, wid, "genuine")
                    sample_log.append([wid, "genuine", path])
            else:
                gen_processed.append(raw_img)

        forg_processed = []
        for i in range(len(forged)):
            raw_img = forged[i]
            if i in forg_idx:
                clahe_img = generator.preprocess_image_from_array(raw_img)
                forg_processed.append(clahe_img)

                if i < 1:
                    path = save_clahe_comparison(raw_img, clahe_img, save_dir, wid, "forged")
                    sample_log.append([wid, "forged", path])
            else:
                forg_processed.append(raw_img)

        processed.extend(gen_processed + forg_processed)

    # Save CSV log
    log_path = os.path.join(save_dir, "sample_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Writer_ID", "Class", "Image_Path"])
        writer.writerows(sample_log)

    print(f"📝 CLAHE sample log saved to: {log_path}")
    return np.array(processed)

def save_clahe_comparison(raw_img, clahe_img, base_dir, writer_id, label):
    # De-normalize if image is in [-1, 1]
    def denorm(img):
        if np.max(img) <= 1.0:
            img = ((img * 0.5) + 0.5) * 255
        return img.astype(np.uint8)

    raw_img = denorm(raw_img)
    clahe_img = denorm(clahe_img)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(raw_img)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(clahe_img)
    axs[1].set_title("CLAHE Enhanced")
    axs[1].axis("off")

    writer_folder = os.path.join(base_dir, f"writer_{writer_id:03d}")
    os.makedirs(writer_folder, exist_ok=True)
    save_path = os.path.join(writer_folder, f"{label}_sample.jpg")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

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

# --- Dataset Configuration ---
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

IMG_SHAPE = (155, 220, 3)
EMBEDDING_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 0.7


# --- Storage ---
balanced_embeddings = {}

# --- CLAHE Preprocessing Only ---
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Preprocessing {dataset_name} ---")
    try:
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=BATCH_SIZE,
        )

        images, class_labels = generator.get_all_data_with_labels()
        _, writer_ids = generator.get_all_data_with_writer_ids()

        # Apply CLAHE to 50% genuine + 50% forged
        X_clahe = apply_partial_clahe_per_writer(
            generator,
            images,
            class_labels,
            writer_ids,
            save_dir=f"clahe_samples/{dataset_name}"
        )
        print("🎨 CLAHE partially applied.")

        # Save processed data
        save_dir = "clahe_data"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{dataset_name}_X.npy"), X_clahe)
        np.save(os.path.join(save_dir, f"{dataset_name}_y.npy"), class_labels)
        np.save(os.path.join(save_dir, f"{dataset_name}_writer_ids.npy"), writer_ids)

    except Exception as e:
        print(f"❌ Error in preprocessing {dataset_name}: {e}")
        continue

# --- Triplet Loss + Hard Negative Mining Training ---
for dataset_name in datasets.keys():
    print(f"\n--- Training Triplet Model for {dataset_name} ---")
    try:
        # Load preprocessed CLAHE-enhanced data
        images = np.load(f"clahe_data/{dataset_name}_X.npy")
        labels = np.load(f"clahe_data/{dataset_name}_y.npy")
        writer_ids = np.load(f"clahe_data/{dataset_name}_writer_ids.npy")

        # Rebuild generator (for later use like test data, not triplet mining)
        dataset_config = datasets[dataset_name]
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=BATCH_SIZE,
        )

        # Create base network
        base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=EMBEDDING_SIZE)

        # ✅ Generate triplets using CLAHE-enhanced images
        anchor_list, positive_list, negative_list = generator.generate_hard_mined_triplets(
        base_network=base_network,
        batch_size=BATCH_SIZE
        )

        if len(anchor_list) == 0:
            print("⚠ No triplets generated. Skipping training.")
            continue

        # Convert lists to numpy arrays
        anc = np.array(anchor_list)
        pos = np.array(positive_list)
        neg = np.array(negative_list)

        # Prepare training data
        train_data = tf.data.Dataset.from_tensor_slices(
            ({"input_anchor": anc, "input_positive": pos, "input_negative": neg}, np.zeros(len(anc)))
        ).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        # Build and train model
        triplet_model = create_triplet_network_from_existing_base(base_network)
        triplet_model.compile(optimizer=RMSprop(learning_rate=0.001), loss=get_triplet_loss(margin=MARGIN))

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)

        print("🚀 Training started...")
        history = triplet_model.fit(train_data, epochs=EPOCHS, verbose=1, callbacks=[early_stop])

        # Save model weights
        base_network.save_weights(f"{dataset_name}_ENHANCED_base_network.weights.h5")

        # Save entire model (architecture + weights)
        triplet_model.save(f"{dataset_name}_ENHANCED_triplet_model.h5")


    except Exception as e:
        print(f"❌ Training failed for {dataset_name}")
        traceback.print_exc()
        continue
        
    # === Evaluation After Training (Enhanced Model) ===
    print(f"\n📊 Starting SOP Evaluation for {dataset_name}")

    # 1. Get test data (clean + noisy)
    test_images, test_labels = generator.get_unbatched_data()
    clean_imgs, clean_labels = test_images, test_labels
    noisy_imgs, _ = generator.get_unbatched_data(noisy=True)
    noisy_imgs = np.array(noisy_imgs)

    if noisy_imgs.ndim != 4:
        raise ValueError("❌ Noisy images must be a 4D tensor (batch, height, width, channels)")

    # 2. Extract embeddings
    embeddings = base_network.predict(clean_imgs, verbose=0)
    clean_emb = embeddings
    noisy_emb = base_network.predict(noisy_imgs, verbose=0)

    # 3. SOP1: Distance Evaluation
    genuine_d, forged_d, distances, binary_labels = [], [], [], []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            label_i = test_labels[i]
            label_j = test_labels[j]

            if label_i == 0 and label_j == 0:
                genuine_d.append(dist)
            elif (label_i == 0 and label_j == 1) or (label_i == 1 and label_j == 0):
                forged_d.append(dist)

    distances = genuine_d + forged_d
    binary_labels = [1] * len(genuine_d) + [0] * len(forged_d)

    # 4. Use F1-optimized threshold
    best_threshold, best_f1 = compute_threshold_f1(distances, binary_labels)
    pred_labels = (np.array(distances) <= best_threshold).astype(int)
    acc = accuracy_score(binary_labels, pred_labels)
    
    # ✅ Save Classification Report CSV
    classification_df = pd.DataFrame({
        "Index": np.arange(len(distances)),
        "Distance": distances,
        "True_Label": binary_labels,
        "Predicted_Label": pred_labels,
        "Result": ["Correct" if t == p else "Incorrect" for t, p in zip(binary_labels, pred_labels)]
    })

    csv_path = f"{dataset_name}_classification_report.csv"
    classification_df.to_csv(csv_path, index=False)
    print(f"📝 classification report saved to {csv_path}")

    # SOP1 Metrics
    sop1_metrics = evaluate_sop1(
        genuine_d=genuine_d,
        forged_d=forged_d,
        binary_labels=binary_labels,
        distances=distances,
        threshold=best_threshold
    )
    sop1_metrics = add_extended_sop1_metrics(
        sop1_metrics,
        binary_labels=binary_labels,
        distances=distances,
        threshold=best_threshold
    )

    # FAR vs FRR Bar Chart
    plot_far_frr_bar_chart(
        roc_far=sop1_metrics['SOP1_FAR'],
        roc_frr=sop1_metrics['SOP1_FRR'],
        dataset_name=dataset_name,
        save_path=f"{dataset_name}_ENHANCED_FAR_FRR_BarChart.png"
    )

    # 5. SOP2: Embedding Space Quality
    sop2_metrics = evaluate_sop2(embeddings, test_labels)

    # 6. SOP3: Noise Robustness
    try:
        psnr_values = [calculate_psnr(c, n) for c, n in zip(clean_imgs, noisy_imgs)]

        sop3_metrics = evaluate_sop3(
            clean_emb=clean_emb,
            noisy_emb=noisy_emb,
            clean_labels=clean_labels,
            threshold=best_threshold
        )
        sop3_metrics['SOP3_Mean_PSNR'] = np.mean(psnr_values)
    except Exception as e:
        print(f"⚠️ SOP3 evaluation failed: {e}")
        sop3_metrics = {k: -1 for k in ['SOP3_Mean_PSNR', 'SOP3_Accuracy_Drop', 'SOP3_Mean_Shift', 'SOP3_Max_Shift']}

    # === Save Enhanced SOP Results ===
    try:
        results = []
        results.append({
            "Dataset": dataset_name,    
            **sop1_metrics,
            **sop2_metrics,
            **sop3_metrics,
            "Accuracy": acc,
            "F1_Score": best_f1
        })
        
        # Save to CSV
        df_path = f"{dataset_name}_enhanced_sop_metrics.csv"
        pd.DataFrame(results).to_csv(df_path, index=False)
        print(f"✅ Saved metrics to {df_path}")

        # Save text summary
        txt_path = f"{dataset_name}_enhanced_results.txt"
        with open(txt_path, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {best_f1:.4f}\n")
            f.write(f"Threshold: {best_threshold:.4f}\n")
            f.write(f"FAR: {sop1_metrics['SOP1_FAR']:.4f}\n")
            f.write(f"FRR: {sop1_metrics['SOP1_FRR']:.4f}\n")
            f.write(f"SOP2 Silhouette: {sop2_metrics.get('SOP2_Silhouette', -1):.4f}\n")
            f.write(f"SOP3 PSNR: {sop3_metrics.get('SOP3_Mean_PSNR', -1):.4f}\n")
            f.write(f"SOP3 Accuracy Drop: {sop3_metrics.get('SOP3_Accuracy_Drop', -1):.4f}\n")
        print(f"📄 Summary saved to {txt_path}")

    except Exception as e:
        print(f"❌ Failed to save metrics: {e}")
    
    # --- Generate reference embeddings (once) ---
    try:
        print(f"\n📦 Generating reference embeddings for {dataset_name}...")

        base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=EMBEDDING_SIZE)
        base_network.load_weights(f"{dataset_name}_ENHANCED_base_network.weights.h5")

        reference_embeddings = []
        reference_labels = []

        for writer_entry in tqdm(generator.train_writers, desc="📥 Embedding genuine references"):
            if isinstance(writer_entry, tuple):
                dataset_path, writer_id = writer_entry
            else:
                dataset_path = dataset_config["path"]
                writer_id = writer_entry

            writer_path = os.path.join(dataset_path, f"writer_{writer_id:03d}")
            genuine_path = os.path.join(writer_path, "genuine")
            if not os.path.exists(genuine_path):
                print(f"⚠️ Skipping writer {writer_id}: path does not exist.")
                continue

            images = [
                os.path.join(genuine_path, f)
                for f in os.listdir(genuine_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            for img_path in images:
                try:
                    img = generator.preprocess_image(img_path)
                    emb = base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                    reference_embeddings.append(emb)
                    reference_labels.append(writer_id)
                except Exception as embed_error:
                    print(f"⚠️ Failed to process image: {img_path}")
                    traceback.print_exc()
                    continue

        reference_embeddings = np.array(reference_embeddings)
        reference_labels = np.array(reference_labels)

        np.save(f"{dataset_name}_ref_embs.npy", reference_embeddings)
        np.save(f"{dataset_name}_ref_labels.npy", reference_labels)

        print(f"✅ Saved {dataset_name}_ref_embs.npy and _ref_labels.npy")

    except Exception as e:
        print(f"❌ Reference embedding failed for {dataset_name}")
        traceback.print_exc()
        continue
