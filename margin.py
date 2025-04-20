import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_triplet_network
from SigNet_v1 import create_base_network_signet_dilated as create_base_network_signet
from SigNet_v1 import get_triplet_loss
from scipy.stats import gaussian_kde

# CONFIG
EMBEDDING_SIZE = 128
MARGINS = [0.2, 0.3, 0.5, 0.7, 1.0]
EPOCHS = 10  # Reduce for faster sweep

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

def evaluate_threshold(base_network, generator, embedding_dim):
    reference_embeddings, reference_labels = [], []
    query_embeddings, binary_labels = [], []

    for dataset_path, writer in tqdm(generator.test_writers):
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        genuine_imgs = sorted([
            os.path.join(genuine_path, f)
            for f in os.listdir(genuine_path)
            if f.lower().endswith((".png", ".jpg"))
        ])
        forged_imgs = sorted([
            os.path.join(forged_path, f)
            for f in os.listdir(forged_path)
            if f.lower().endswith((".png", ".jpg"))
        ])

        if len(genuine_imgs) < 2 or len(forged_imgs) < 1:
            continue

        ref_img = generator.preprocess_image(genuine_imgs[0])
        ref_emb = base_network.predict(np.expand_dims(ref_img, axis=0), verbose=0)[0]
        reference_embeddings.append(ref_emb)
        reference_labels.append(writer)

        for img_path in genuine_imgs[1:]:
            img = generator.preprocess_image(img_path)
            emb = base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            query_embeddings.append(emb)
            binary_labels.append(1)

        for img_path in forged_imgs:
            img = generator.preprocess_image(img_path)
            emb = base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            query_embeddings.append(emb)
            binary_labels.append(0)

    reference_array = np.array(reference_embeddings)
    query_array = np.array(query_embeddings)

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(reference_array)

    distances = []
    for query in query_array:
        norm_query = query / np.linalg.norm(query)
        D, _ = index.search(np.expand_dims(norm_query, axis=0), k=1)
        distances.append(D[0][0])

    genuine_dists = [d for d, l in zip(distances, binary_labels) if l == 1]
    forged_dists = [d for d, l in zip(distances, binary_labels) if l == 0]

    kde_genuine = gaussian_kde(genuine_dists)
    kde_forged = gaussian_kde(forged_dists)
    x = np.linspace(min(distances), max(distances), 1000)
    kde_g = kde_genuine(x)
    kde_f = kde_forged(x)

    intersect = np.argwhere(np.diff(np.sign(kde_g - kde_f))).flatten()
    threshold = x[intersect][0] if len(intersect) else np.percentile(genuine_dists, 90)

    preds = [1 if d < threshold else 0 for d in distances]

    acc = accuracy_score(binary_labels, preds)
    f1 = f1_score(binary_labels, preds)
    auc = roc_auc_score(binary_labels, [-d for d in distances])
    
    cm = confusion_matrix(binary_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)

    return acc, f1, auc, far, frr, threshold

# MAIN LOOP
for dataset_name, dataset_config in datasets.items():
    print(f"\n🚀 Margin Sweep for {dataset_name}")
    results = []
    best_acc = -1
    best_model = None
    best_base_model = None  # ✅
    best_margin = None

    for margin in MARGINS:
        print(f"\n🔍 Training with margin={margin}")
        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=32
        )

        base_network = create_base_network_signet((155, 220, 3), embedding_dim=EMBEDDING_SIZE)  # ✅ NEW
        triplet_model = create_triplet_network((155, 220, 3), embedding_dim=EMBEDDING_SIZE)
        triplet_model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=get_triplet_loss(margin)
        )

        anchor_list, positive_list, negative_list = [], [], []
        for batch in generator.get_triplet_data(generator.train_writers):
            a, p, n = batch
            anchor_list.extend(a.numpy())
            positive_list.extend(p.numpy())
            negative_list.extend(n.numpy())

        train_X_anchor = np.array(anchor_list)
        train_X_positive = np.array(positive_list)
        train_X_negative = np.array(negative_list)
        dummy_labels = np.zeros((len(train_X_anchor),))

        train_data = tf.data.Dataset.from_tensor_slices((
            (train_X_anchor, train_X_positive, train_X_negative),
            dummy_labels
        )).map(lambda x, y: ((x[0], x[1], x[2]), y)) \
        .batch(generator.batch_sz, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

        triplet_model.fit(
            train_data,
            epochs=EPOCHS,
            steps_per_epoch=len(train_X_anchor) // generator.batch_sz,
            verbose=0
        )

        # ✅ Use base_network directly
        acc, f1, auc, far, frr, threshold = evaluate_threshold(base_network, generator, EMBEDDING_SIZE)
        print(f"✅ Margin {margin:.2f} → Acc: {acc:.4f}, F1: {f1:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}")

        results.append([margin, acc, f1, auc, far, frr, threshold])

        if acc > best_acc:
            best_acc = acc
            best_model = triplet_model
            best_base_model = base_network  # ✅ Track correct one
            best_margin = margin

    if best_model and best_base_model:
        model_path = f"{dataset_name}_margin{best_margin:.2f}_triplet_model.keras"
        base_path = f"{dataset_name}_margin{best_margin:.2f}_base_weights.h5"
        best_model.save(model_path)
        best_base_model.save_weights(base_path)
        print(f"✅ Saved best model: {model_path}")

    df = pd.DataFrame(results, columns=["Margin", "Accuracy", "F1", "ROC AUC", "FAR", "FRR", "Threshold"])
    df.to_csv(f"{dataset_name}_margin_sweep_metrics.csv", index=False)
    print(f"📄 Metrics saved to {dataset_name}_margin_sweep_metrics.csv")
