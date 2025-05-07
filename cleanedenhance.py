import os
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import time
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

# too many values to unpack (expected 2)

# --- Storage ---
balanced_embeddings = {}

# --- CLAHE + Embedding + SMOTE ---
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Preprocessing {dataset_name} ---")
    try:
        dataset_config = datasets[dataset_name]
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=BATCH_SIZE,
        )
        

        # --- Merged SignatureDataGenerator ---
        # generator = SignatureDataGenerator(
        #     dataset=merged_train_config,
        #     img_height=155,
        #     img_width=220,
        #     batch_sz=BATCH_SIZE
        # )

        # # ✅ Set this to label outputs and saved weights
        # dataset_name = "Merged"


        generator.save_dataset_to_csv(f"{dataset_name}_signature_dataset.csv")
        generator.visualize_clahe_effect(output_dir=f"CLAHE_Comparison_{dataset_name}")

        base_network = create_base_network_signet((155, 220, 3), embedding_dim=EMBEDDING_SIZE)
        all_images, all_labels = generator.get_all_data_with_labels()  # CLAHE is applied here
        embeddings = base_network.predict(all_images)
        print(f"🔎 Class distribution before SMOTE: {Counter(all_labels)}")

        if dataset_name == "CEDAR":
            print("🧼 SMOTE skipped for CEDAR")
            balanced_embeddings[dataset_name] = (embeddings, all_labels)
        else:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(embeddings, all_labels)
            print(f"✅ SMOTE applied. New distribution: {Counter(y_res)}")
            balanced_embeddings[dataset_name] = (X_res, y_res)

    except Exception as e:
        print(f"❌ Error in preprocessing {dataset_name}: {e}")
        continue


# --- Training Using Triplet Loss + HNM  ---
for dataset_name, (embeddings, labels) in balanced_embeddings.items():
    print(f"\n--- Training Triplet Model for {dataset_name} ---")
    
    try:
        # --- Generator ---
        dataset_config = datasets[dataset_name]
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=BATCH_SIZE,
        )

        base_network = create_base_network_signet(IMG_SHAPE, embedding_dim=EMBEDDING_SIZE)

        # 🧠 Generate hard-mined triplets
        anchor_imgs, positive_imgs, negative_imgs = generator.generate_hard_mined_triplets(base_network)
        print(f"✅ Triplets generated: {len(anchor_imgs)}")

        dummy_labels = np.zeros((len(anchor_imgs),))

        train_data = tf.data.Dataset.from_tensor_slices((
            (anchor_imgs, positive_imgs, negative_imgs),
            dummy_labels
        )).map(lambda x, y: ((x[0], x[1], x[2]), y)) \
         .batch(BATCH_SIZE, drop_remainder=True) \
         .prefetch(tf.data.AUTOTUNE)

        # 🧱 Build and train model
        triplet_model = create_triplet_network_from_existing_base(base_network)

        triplet_model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss=get_triplet_loss(margin=MARGIN)
        )

        history = triplet_model.fit(
            train_data,
            epochs=EPOCHS,
            verbose=1,
        )

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

        def compute_far_frr(y_true, y_pred):
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            return far, frr

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

        # ✅ Save model weights
        base_network.save_weights(f"{dataset_name}_base_network.weights.h5")

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

