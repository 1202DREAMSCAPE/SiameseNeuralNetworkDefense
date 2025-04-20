import os
import numpy as np
import cv2
from collections import Counter
from imblearn.over_sampling import SMOTE
import faiss
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)
from scipy.stats import gaussian_kde

def evaluate_threshold(base_model, generator, embedding_size):
    try:
        query_embeddings, binary_labels = [], []
        reference_embeddings, reference_labels = [], []

        for dataset_path, writer in generator.test_writers:
            writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
            genuine_path = os.path.join(writer_path, "genuine")
            forged_path = os.path.join(writer_path, "forged")

            genuine_files = sorted([os.path.join(genuine_path, f) for f in os.listdir(genuine_path) if f.lower().endswith(("png", "jpg"))])
            forged_files = sorted([os.path.join(forged_path, f) for f in os.listdir(forged_path) if f.lower().endswith(("png", "jpg"))])

            if len(genuine_files) < 2 or len(forged_files) < 1:
                continue

            reference_img = generator.preprocess_image(genuine_files[0])
            ref_emb = base_model.predict(np.expand_dims(reference_img, axis=0), verbose=0)[0]
            reference_embeddings.append(ref_emb)
            reference_labels.append(writer)

            for img_path in genuine_files[1:]:
                img = generator.preprocess_image(img_path)
                emb = base_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                query_embeddings.append(emb)
                binary_labels.append(1)

            for img_path in forged_files:
                img = generator.preprocess_image(img_path)
                emb = base_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                query_embeddings.append(emb)
                binary_labels.append(0)

        if len(reference_embeddings) == 0:
            return None

        index = faiss.IndexFlatL2(embedding_size)
        index.add(np.array(reference_embeddings))

        distances = []
        for emb in query_embeddings:
            emb_norm = emb / np.linalg.norm(emb)
            D, _ = index.search(np.expand_dims(emb_norm, axis=0), k=1)
            distances.append(D[0][0])

        # KDE for threshold
        genuine_dists = [d for d, l in zip(distances, binary_labels) if l == 1]
        forged_dists = [d for d, l in zip(distances, binary_labels) if l == 0]

        kde_gen = gaussian_kde(genuine_dists)
        kde_forg = gaussian_kde(forged_dists)

        x = np.linspace(min(min(genuine_dists), min(forged_dists)), max(max(genuine_dists), max(forged_dists)), 1000)
        gen_density = kde_gen(x)
        forg_density = kde_forg(x)

        intersection_idx = np.argwhere(np.diff(np.sign(gen_density - forg_density))).flatten()
        threshold = x[intersection_idx][0] if len(intersection_idx) > 0 else np.percentile(genuine_dists, 90)

        y_pred = [1 if d < threshold else 0 for d in distances]

        acc = accuracy_score(binary_labels, y_pred)
        f1 = f1_score(binary_labels, y_pred)
        auc = roc_auc_score(binary_labels, [-d for d in distances])
        far = sum((np.array(y_pred) == 1) & (np.array(binary_labels) == 0)) / max(sum(np.array(binary_labels) == 0), 1)
        frr = sum((np.array(y_pred) == 0) & (np.array(binary_labels) == 1)) / max(sum(np.array(binary_labels) == 1), 1)

        return acc, f1, auc, far, frr, threshold

    except Exception as e:
        print(f"❌ Error in evaluate_threshold: {e}")
        return None
        
# Add noise to an image
def add_noise_to_image(img):
    h, w, _ = img.shape
    img = cv2.resize(img, (w // 2, h // 2))
    img = cv2.resize(img, (w, h))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
    noisy_img = np.clip(img + noise, -1.0, 1.0)
    return noisy_img.astype(np.float32)

# Apply SMOTE separately to each dataset
def apply_smote_per_dataset(datasets):
    balanced_embeddings = []
    balanced_labels = []

    for dataset_name, (X, y) in datasets.items():
        class_counts = Counter(y)
        print(f"\n🔍 Class Distribution for {dataset_name} Before SMOTE:")
        for label, count in class_counts.items():
            print(f"   - Class {label}: {count} samples")

        if min(class_counts.values()) < max(class_counts.values()):
            print(f"⚠ Imbalance detected in {dataset_name}. Applying SMOTE...")
            try:
                smote = SMOTE(sampling_strategy="auto", random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"✅ Class Distribution for {dataset_name} After SMOTE:")
                for label, count in Counter(y_resampled).items():
                    print(f"   - Class {label}: {count} samples")
                balanced_embeddings.append(X_resampled)
                balanced_labels.append(y_resampled)
            except Exception as e:
                print(f"❌ SMOTE failed for {dataset_name}: {e}")
                balanced_embeddings.append(X)
                balanced_labels.append(y)
        else:
            print(f"⚠ No imbalance detected for {dataset_name}. Skipping SMOTE.")
            balanced_embeddings.append(X)
            balanced_labels.append(y)

    X_final = np.vstack(balanced_embeddings)
    y_final = np.hstack(balanced_labels)

    print("\n🌟 Combined Balanced Dataset:")
    print(f"   - Total Samples: {len(y_final)}")
    print(f"   - Class Distribution: {Counter(y_final)}")

    return X_final, y_final

# FAISS Indexing
def build_faiss_index(embeddings, dim):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(index, query_embedding, k=1):
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Hard Negative Mining
def find_hard_negatives(distances, labels, threshold, predictions):
    return [i for i, (dist, label, pred) in enumerate(zip(distances, labels, predictions)) if label == 0 and pred == 1]

def compute_hard_negative_metrics(y_true, y_pred, hard_negative_indices):
    hard_y_true = [y_true[i] for i in hard_negative_indices]
    hard_y_pred = [y_pred[i] for i in hard_negative_indices]

    if not hard_y_true:
        print("No hard negatives found.")
        return 0, 0, 0

    forged_total = sum(1 for label in y_true if label == 0)
    hard_negative_ratio = len(hard_negative_indices) / forged_total if forged_total else 0

    tp = sum(1 for t, p in zip(hard_y_true, hard_y_pred) if t == 0 and p == 1)
    fp = sum(1 for t, p in zip(hard_y_true, hard_y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(hard_y_true, hard_y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return hard_negative_ratio, precision, recall

# Evaluation
def evaluate_with_faiss(base_network, index, test_images, test_labels, threshold):
    y_true, y_pred = [], []
    query_embeds = base_network.predict(test_images)

    for i, query in enumerate(query_embeds):
        label = test_labels[i]
        query_norm = query / np.linalg.norm(query)
        D, _ = index.search(np.expand_dims(query_norm, axis=0), k=1)
        distance = D[0][0]
        prediction = 1 if distance <= threshold else 0
        y_true.append(label)
        y_pred.append(prediction)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1
