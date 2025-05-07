import os
import numpy as np
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
    apply_smote_per_dataset,
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
from SigNet_v1 import create_base_network_signet_dilated as create_base_network_signet
from SigNet_v1 import create_triplet_network_from_existing_base
from imblearn.over_sampling import SMOTE
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

    step = 0.01
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

# --- Merged Dataset for Training ---
merged_train_config = {
    "Merged": {
        "path": "",  # left blank; handled per writer
        "train_writers": []
    }
}

# Combine all train writers from each dataset
for dataset_name, config in datasets.items():
    path = config["path"]
    for writer in config["train_writers"]:
        merged_train_config["Merged"]["train_writers"].append({
            "path": path,
            "writer": writer
        })

IMG_SHAPE = (155, 220, 3)
EMBEDDING_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 0.7


# --- Storage ---
balanced_embeddings = {}

# --- SMOTE → Partial CLAHE → Embedding + CSV Logging ---
for dataset_name, dataset_config in datasets.items():
    smote_folder = f"smote_{dataset_name}"
    os.makedirs(smote_folder, exist_ok=True)

    print(f"\n--- Preprocessing {dataset_name} ---")
    try:
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=BATCH_SIZE,
        )

        raw_images, class_labels = generator.get_all_data_with_labels()
        _, writer_ids = generator.get_all_data_with_writer_ids()

        smote = SMOTE(random_state=42)
        Xb_list, yb_list, wid_list = [], [], []

        # For CSV logging
        csv_rows = []
        csv_header = ["Dataset", "Writer_ID", "Before_Genuine", "Before_Forged", "After_Genuine", "After_Forged", "Synthetic_Added"]
        
        pre_smote_csv = os.path.join(smote_folder, f"{dataset_name}_pre_smote_distribution.csv")
        with open(pre_smote_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Writer_ID", "Genuine_Before", "Forged_Before"])

            for wid in np.unique(writer_ids):
                mask = (writer_ids == wid)
                _, yw = raw_images[mask], class_labels[mask]
                counts = Counter(yw)
                genuine = counts.get(1, 0)
                forged = counts.get(0, 0)
                writer.writerow([wid, genuine, forged])

        for wid in np.unique(writer_ids):
            mask = (writer_ids == wid)
            Xw, yw = raw_images[mask], class_labels[mask]
            before_count = Counter(yw)

            if len(np.unique(yw)) > 1:
                Xw_flat = Xw.reshape(len(Xw), -1)
                X_res, y_res = smote.fit_resample(Xw_flat, yw)
                X_res = X_res.reshape((-1, 155, 220, 3))
            else:
                X_res, y_res = Xw, yw

            after_count = Counter(y_res)
            synth_added = len(y_res) - len(yw)

            # Log writer SMOTE info
            csv_rows.append([
                dataset_name,
                wid,
                before_count.get(1, 0),
                before_count.get(0, 0),
                after_count.get(1, 0),
                after_count.get(0, 0),
                synth_added
            ])

            Xb_list.append(X_res)
            yb_list.append(y_res)
            wid_list.append(np.full(len(X_res), wid))

        X_bal = np.concatenate(Xb_list)
        y_bal = np.concatenate(yb_list)
        wids_bal = np.concatenate(wid_list)

        print(f"✅ SMOTE done. Class dist: {Counter(y_bal)}")

        # Save to CSV
        post_smote_csv = os.path.join(smote_folder, f"{dataset_name}_post_smote_distribution.csv")
        with open(post_smote_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Writer_ID", "Genuine_After", "Forged_After", "Synthetic_Added"])
            writer.writerows(csv_rows)

        # Apply CLAHE to 50% genuine + 50% forged
        X_clahe = apply_partial_clahe_per_writer(
            generator,
            X_bal,
            y_bal,
            wids_bal,
            save_dir=f"clahe_samples/{dataset_name}"
        )
        print("🎨 CLAHE partially applied.")

        balanced_embeddings[dataset_name] = (X_clahe, y_bal, wids_bal)

    except Exception as e:
        print(f"❌ Error in preprocessing {dataset_name}: {e}")
        continue

# --- Triplet Loss + Hard Negative Mining Training ---
# Load processed images from balanced_embeddings
for dataset_name, (images, labels, writer_ids) in balanced_embeddings.items():
    print(f"\n--- Training Triplet Model for {dataset_name} ---")
    try:
        dataset_config = datasets[dataset_name]
        
        # Create generator manually for that dataset (use dummy path)
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=BATCH_SIZE,
        )

        # ✅ Use your generator's hard-mined triplet method
        base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=EMBEDDING_SIZE)
        anc, pos, neg = generator.generate_hard_mined_triplets(base_network)

        dummy_labels = np.zeros((len(anc),))
        train_data = tf.data.Dataset.from_tensor_slices(((anc, pos, neg), dummy_labels)) \
            .map(lambda x, y: ((x[0], x[1], x[2]), y)) \
            .batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        # Build and train model
        triplet_model = create_triplet_network_from_existing_base(base_network)
        triplet_model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss=get_triplet_loss(margin=MARGIN)
        )

        print("🚀 Training started...")
        history = triplet_model.fit(train_data, epochs=EPOCHS, verbose=1)

        # Save model weights
        base_network.save_weights(f"{dataset_name}_base_network.weights.h5")
        
        # --- Evaluation ---
        test_imgs, test_labels = generator.get_test_data()
        test_embeddings = base_network.predict(test_imgs)

        # ✅ Manual distance + label generation
        distances = []
        binary_labels = []

        for i in range(len(test_labels)):
            for j in range(i + 1, len(test_labels)):
                dist = np.linalg.norm(test_embeddings[i] - test_embeddings[j])
                label = 1 if test_labels[i] == test_labels[j] else 0
                distances.append(dist)
                binary_labels.append(label)

        distances = np.array(distances)
        binary_labels = np.array(binary_labels)

        acc, threshold = compute_accuracy_roc(distances, binary_labels)
        pred_labels = (distances < threshold).astype(int)
        f1 = f1_score(binary_labels, pred_labels)
        far, frr = compute_far_frr(binary_labels, pred_labels)

        print(f"📊 Accuracy: {acc:.4f} | F1: {f1:.4f} | Threshold: {threshold:.4f} | FAR: {far:.4f} | FRR: {frr:.4f}")

        # --- Log to txt file ---
        with open(f"{dataset_name}_results.txt", "w") as f:
            f.write(
                f"Dataset: {dataset_name}\n"
                f"Accuracy: {acc:.4f}\n"
                f"F1 Score: {f1:.4f}\n"
                f"Threshold: {threshold:.4f}\n"
                f"FAR: {far:.4f}\n"
                f"FRR: {frr:.4f}\n"
            )

        # --- Generate reference embeddings (once) ---
        print(f"\n📦 Generating reference embeddings for {dataset_name}...")

        base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=EMBEDDING_SIZE)
        base_network.load_weights(f"{dataset_name}_base_network.weights.h5")

        reference_embeddings = []
        reference_labels = []

        for writer_id in tqdm(generator.train_writers, desc="📥 Embedding genuine references"):
            writer_path = os.path.join(dataset_config["path"], f"writer_{writer_id:03d}")
            genuine_path = os.path.join(writer_path, "genuine")
            images = [
                os.path.join(genuine_path, f)
                for f in os.listdir(genuine_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            for img_path in images:
                img = generator.preprocess_image_simple(img_path)
                emb = base_network.predict(np.expand_dims(img, axis=0))[0]
                reference_embeddings.append(emb)
                reference_labels.append(writer_id)

        reference_embeddings = np.array(reference_embeddings)
        reference_labels = np.array(reference_labels)

        np.save(f"{dataset_name}_ref_embs.npy", reference_embeddings)
        np.save(f"{dataset_name}_ref_labels.npy", reference_labels)

        print(f"✅ Saved {dataset_name}_ref_embs.npy and _ref_labels.npy")

    except Exception as e:
        print(f"❌ Training failed for {dataset_name}: {e}")
        continue

