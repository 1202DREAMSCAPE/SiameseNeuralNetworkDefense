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
from SignatureDataGenerator import SignatureDataGenerator, get_all_data_with_writer_ids
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


# --- Storage ---
balanced_embeddings = {}

# --- Merged SignatureDataGenerator ---
        # generator = SignatureDataGenerator(
        #     dataset=merged_train_config,
        #     img_height=155,
        #     img_width=220,
        #     batch_sz=BATCH_SIZE
        # )

        # # ✅ Set this to label outputs and saved weights
        # dataset_name = "Merged"

# ------------------------------------------------------------------
# 0. configuration
# ------------------------------------------------------------------
IMG_SHAPE       = (155, 220, 3)
EMBEDDING_SIZE  = 128
BATCH_SIZE      = 32
EPOCHS_CNN      = 20      # full‑CNN training
EPOCHS_HEAD     = 20      # projection‑head training
MARGIN          = 0.7
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# 1.  helper functions  (add ONCE, after your imports)
# ------------------------------------------------------------------
def build_triplets_from_embeddings(X, y, k_hard=1):
    anchors, positives, negatives = [], [], []
    for idx, (emb, lbl) in enumerate(zip(X, y)):
        same = np.where(y == lbl)[0]; same = same[same != idx]
        if len(same) == 0:  continue
        pos_idx = np.random.choice(same)
        diff = np.where(y != lbl)[0]
        dists = np.linalg.norm(X[diff] - emb, axis=1)
        for hard in np.argsort(dists)[:k_hard]:
            anchors.append(emb)
            positives.append(X[pos_idx])
            negatives.append(X[diff[hard]])
    return np.array(anchors), np.array(positives), np.array(negatives)


def build_embedding_triplet_head(dim):
    ia = tf.keras.Input(shape=(dim,)); ip = tf.keras.Input(shape=(dim,)); in_ = tf.keras.Input(shape=(dim,))
    proj = tf.keras.layers.Dense(dim, use_bias=False, name="proj")
    return tf.keras.Model([ia, ip, in_], [proj(ia), proj(ip), proj(in_)], name="triplet_head")
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# 2.  loop over datasets
# ------------------------------------------------------------------
for dname, dcfg in datasets.items():
    print(f"\n================  {dname}  ================")
    try:
        gen = SignatureDataGenerator(
            dataset={dname: dcfg},
            img_height=155, img_width=220, batch_sz=BATCH_SIZE)

        # ----------------------------------------------------------
        # 2‑A. FIRST‑TIME CNN TRAINING  (image triplets)
        # ----------------------------------------------------------
        print(f"➡ Training base ({EPOCHS_CNN} epochs)…")

        base = create_base_network_signet(IMG_SHAPE, embedding_dim=EMBEDDING_SIZE)

        anc, pos, neg = gen.generate_hard_mined_triplets(base)
        anc = np.asarray(anc, dtype=np.float32)
        pos = np.asarray(pos, dtype=np.float32)
        neg = np.asarray(neg, dtype=np.float32)

        # optional: shuffle
        idx = np.random.permutation(len(anc))
        anc, pos, neg = anc[idx], pos[idx], neg[idx]

        dummy = np.zeros((len(anc), 1), dtype=np.float32)   # Keras-compatible y

        triplet_model = create_triplet_network_from_existing_base(base)  # returns 3-input model
        triplet_model.compile(optimizer=RMSprop(1e-3),
                            loss=get_triplet_loss(margin=MARGIN))

        triplet_model.fit([anc, pos, neg],
                        dummy,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS_CNN,
                        verbose=1)

        # save the trained CNN
        base.save_weights(f"{dname}_base_network.weights.h5")
        print(f"✅  Saved base CNN weights ➜  {dname}_base_network.weights.h5")

        # ----------------------------------------------------------
        # 2‑B.   SMOTE‑BALANCED EMBEDDING‑SPACE HEAD (Option A)
        # ----------------------------------------------------------
        # 2‑B‑1  extract embeddings & writer‑ids
        imgs, writer_ids = gen.get_all_data_with_writer_ids()
        embeds = base.predict(imgs, batch_size=BATCH_SIZE, verbose=0)

        if dname == "CEDAR":          # skip SMOTE if you like
            X_bal, y_bal = embeds, writer_ids
            print("🧼  SMOTE skipped.")
        else:
            sm = SMOTE(random_state=42)
            X_bal, y_bal = sm.fit_resample(embeds, writer_ids)
            print(f"✅  SMOTE applied: {Counter(y_bal)}")

        # 2‑B‑2  triplets in embedding space
        A, P, N = build_triplets_from_embeddings(X_bal, y_bal, k_hard=1)
        print(f"➡  Training projection head ({EPOCHS_HEAD} epochs) …   triplets={len(A)}")

        head = build_embedding_triplet_head(EMBEDDING_SIZE)
        head.compile(optimizer=RMSprop(1e-3),
                     loss=get_triplet_loss(margin=MARGIN))
        head.fit([A, P, N],
                 np.zeros(len(A)),
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS_HEAD,
                 verbose=1)

        head.save_weights(f"{dname}_triplet_head.weights.h5")
        proj_mat = head.get_layer("proj").get_weights()[0]
        np.save(f"{dname}_proj_matrix.npy", proj_mat)
        print(f"✅  Saved projection head & matrix for {dname}")

        # ----------------------------------------------------------
        # 2‑C.   EVALUATION  (raw → CNN → projection)
        # ----------------------------------------------------------
        test_imgs, test_labels = gen.get_unbatched_data()
        raw_emb = base.predict(test_imgs, batch_size=BATCH_SIZE, verbose=0)
        test_emb = raw_emb @ proj_mat

        dists, bin_lab = [], []
        for i in range(len(test_labels)):
            for j in range(i+1, len(test_labels)):
                dists.append(np.linalg.norm(test_emb[i]-test_emb[j]))
                bin_lab.append(int(test_labels[i] == test_labels[j]))
        dists = np.asarray(dists); bin_lab = np.asarray(bin_lab)

        acc, thr = compute_accuracy_roc(dists, bin_lab)
        y_pred = (dists < thr).astype(int)
        f1  = f1_score(bin_lab, y_pred)
        far, frr = compute_far_frr(bin_lab, y_pred)

        print(f"📊  Acc={acc:.4f} | F1={f1:.4f} | Thr={thr:.4f} | FAR={far:.4f} | FRR={frr:.4f}")

        # ----------------------------------------------------------
        # 2‑D. reference embeddings  (genuine only)
        # ----------------------------------------------------------
        ref_emb, ref_lbls = [], []
        for path, w in gen.train_writers:
            gdir = os.path.join(path, f"writer_{w:03d}", "genuine")
            for fn in os.listdir(gdir):
                img = gen.preprocess_image(os.path.join(gdir, fn))
                emb = base.predict(np.expand_dims(img,0), verbose=0)[0]
                ref_emb.append(emb @ proj_mat)
                ref_lbls.append(w)

        np.save(f"{dname}_ref_embs.npy",  np.array(ref_emb))
        np.save(f"{dname}_ref_labels.npy", np.array(ref_lbls))
        print(f"✅  Reference embeddings saved for {dname}")

        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"❌  {dname} failed: {e}")
        continue
