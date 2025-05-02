import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, silhouette_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
from SignatureDataGenerator import SignatureDataGenerator

# Preprocessing
def preprocess_image_simple(img_path, img_height, img_width):
    if not isinstance(img_path, str) or not os.path.exists(img_path):
        return np.zeros((img_height, img_width, 3), dtype=np.float32)
    try:
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((img_height, img_width, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype(np.float32) / 255.0
        return (img - 0.5) / 0.5
    except Exception:
        return np.zeros((img_height, img_width, 3), dtype=np.float32)

SignatureDataGenerator.preprocess_image = lambda self, img_path: preprocess_image_simple(img_path, self.img_height, self.img_width)

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), tf.keras.backend.epsilon()))

def get_contrastive_loss(margin=1.0):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

def create_base_network_signet(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        layers.Conv2D(96, (11,11), activation='relu', strides=(4,4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        layers.ZeroPadding2D((2,2)),
        layers.Conv2D(256, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        layers.Dropout(0.3),
        layers.ZeroPadding2D((1,1)),
        layers.Conv2D(384, (3,3), activation='relu'),
        layers.ZeroPadding2D((1,1)),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
    ])
    return model

def evaluate_sop1(genuine_d, forged_d, binary_labels, distances):
    metrics = {
        'SOP1_Mean_Genuine': np.mean(genuine_d),
        'SOP1_Mean_Forged': np.mean(forged_d),
        'SOP1_Std_Genuine': np.std(genuine_d),
        'SOP1_Std_Forged': np.std(forged_d)
    }
    fpr, tpr, thresholds = roc_curve(binary_labels, distances)
    try:
        eer_func = interp1d(fpr, tpr)
        eer = brentq(lambda x: 1. - x - eer_func(x), 0., 1.)
    except Exception:
        eer = -1
    metrics.update({
        'SOP1_EER': eer,
        'SOP1_AUC_ROC': roc_auc_score(binary_labels, -np.array(distances)),
        'SOP1_Threshold': thresholds[np.argmax(tpr - fpr)]
    })
    hist_gen, bins = np.histogram(genuine_d, bins=30, density=True)
    hist_forg, _ = np.histogram(forged_d, bins=bins, density=True)
    metrics['SOP1_Overlap_Area'] = np.sum(np.minimum(hist_gen, hist_forg)) * (bins[1] - bins[0])
    preds = [1 if d > metrics['SOP1_Threshold'] else 0 for d in distances]
    cm = confusion_matrix(binary_labels, preds)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        metrics['SOP1_FAR'] = FP / (FP + TN) if (FP + TN) != 0 else np.nan
        metrics['SOP1_FRR'] = FN / (FN + TP) if (FN + TP) != 0 else np.nan
    else:
        metrics['SOP1_FAR'], metrics['SOP1_FRR'] = np.nan, np.nan
    return metrics

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

def evaluate_sop3(clean_emb, noisy_emb, clean_labels, threshold):
    shifts = np.linalg.norm(clean_emb - noisy_emb, axis=1)
    dists = np.array([np.min(np.linalg.norm(clean_emb[clean_labels == 0] - e, axis=1)) for e in noisy_emb])
    preds = (dists > threshold).astype(int)
    return {
        'SOP3_Mean_PSNR': np.mean([calculate_psnr(clean, noisy) for clean, noisy in zip(clean_emb, noisy_emb)]),
        'SOP3_Accuracy_Drop': accuracy_score(clean_labels, preds),
        'SOP3_Mean_Shift': np.mean(shifts),
        'SOP3_Max_Shift': np.max(shifts)
    }

# ========== CONFIG ==========
IMG_SHAPE = (155, 220, 3)
BATCH_SIZE = 32
EPOCHS = 40
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

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing {dataset_name}")

    # Data loading
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )
    pairs, labels = generator.generate_pairs()
    pairs, labels = np.array(pairs), np.array(labels).astype(np.float32)

    # Train/val split
    val_split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:val_split], pairs[val_split:]
    train_labels, val_labels = labels[:val_split], labels[val_split:]

    # Model setup
    base_network = create_base_network_signet(IMG_SHAPE)
    input_a = Input(shape=IMG_SHAPE)
    input_b = Input(shape=IMG_SHAPE)
    distance = layers.Lambda(euclidean_distance)([base_network(input_a), base_network(input_b)])
    model = Model([input_a, input_b], distance)
    model.compile(optimizer=RMSprop(0.0001), loss=get_contrastive_loss(MARGIN))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # ========== Training ==========
    start_time = time.time()
    history = model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )
    train_time = time.time() - start_time

    # ========== Embedding Extraction ==========
    test_images, test_labels = generator.get_unbatched_data()
    embeddings = base_network.predict(test_images, verbose=0)

    # ========== SOP 1 Evaluation ==========
    genuine_d, forged_d, distances, binary_labels = [], [], [], []
    for i, emb in enumerate(embeddings):
        dists = [np.linalg.norm(emb - embeddings[j]) for j in range(len(embeddings)) if i != j]
        min_dist = min(dists)
        distances.append(min_dist)
        binary_labels.append(int(test_labels[i]))
        if test_labels[i] == 0:
            genuine_d.append(min_dist)
        else:
            forged_d.append(min_dist)
    sop1_metrics = evaluate_sop1(genuine_d, forged_d, binary_labels, distances)

    # ========== SOP 2 Evaluation ==========
    sop2_metrics = evaluate_sop2(embeddings, test_labels)

    # ========== SOP 3 Evaluation ==========
    try:
        clean_imgs, _ = generator.get_unbatched_data()
        noisy_imgs, _ = generator.get_unbatched_data(noisy=True)
        noisy_emb = base_network.predict(noisy_imgs, verbose=0)
        sop3_metrics = evaluate_sop3(embeddings, noisy_emb, test_labels, sop1_metrics['SOP1_Threshold'])
    except Exception as e:
        print(f"⚠️ SOP 3 failed: {e}")
        sop3_metrics = {k: -1 for k in ['SOP3_Mean_PSNR', 'SOP3_Accuracy_Drop', 'SOP3_Mean_Shift', 'SOP3_Max_Shift']}

    # ========== Collect Results ==========
    acc = accuracy_score(binary_labels, [1 if d > sop1_metrics['SOP1_Threshold'] else 0 for d in distances])
    f1 = f1_score(binary_labels, [1 if d > sop1_metrics['SOP1_Threshold'] else 0 for d in distances])

    results.append({
        "Dataset": dataset_name,
        "Training_Time": train_time,
        **sop1_metrics,
        **sop2_metrics,
        **sop3_metrics,
        "Accuracy": acc,
        "F1_Score": f1
    })

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'{dataset_name}_model.h5')
    model.save(model_path)
    # ========== Save Results to CSV ==========
    # ✅ Save CSV immediately after each dataset completes:
    pd.DataFrame(results).to_csv("SigNet_Baseline_SOP_Results.csv", index=False)
    print(f"✅ Metrics saved to CSV after processing {dataset_name}.")

    # ========== Visualization ==========
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.hist(genuine_d, bins=30, alpha=0.6, label='Genuine')
    plt.hist(forged_d, bins=30, alpha=0.6, label='Forged')
    plt.axvline(sop1_metrics['SOP1_Threshold'], color='r', linestyle='--')
    plt.title(f"{dataset_name} Distance Distribution")
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
    plt.savefig(f"{dataset_name}_baseline_metrics.png")
    plt.close()

    # ====== Loss Curve Plot (Training vs Validation) ======
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} Training Loss Curve (Contrastive Loss)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_baseline_loss_curve.png")
    plt.close()

print(f"📋 Finished evaluation for {dataset_name}. Current CSV updated.\n")

# ========== Cross-Dataset Evaluation ==========
print("\n🔁 Starting Cross-Dataset Evaluation...")
cross_results = []

for train_name in datasets.keys():
    weight_path = f"models/{train_name}_model.h5"
    if not os.path.exists(weight_path):
        print(f"❌ Model not found for {train_name}, skipping.")
        continue

    base_model = create_base_network_signet(IMG_SHAPE)
    base_model.load_weights(weight_path)

    for test_name, config in datasets.items():
        print(f"\n🔍 {train_name} → {test_name}")
        generator = SignatureDataGenerator(
            dataset={test_name: config},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE
        )

        test_images, test_labels = generator.get_unbatched_data()
        embeddings = base_model.predict(test_images, verbose=0)

        # Reference mean embedding of genuine
        genuine_embs = embeddings[np.array(test_labels) == 0]
        ref = np.mean(genuine_embs, axis=0)
        distances = np.linalg.norm(embeddings - ref, axis=1)

        dmax = np.max(distances)
        dmin = np.min(distances)
        nsame = np.sum(test_labels == 0)
        ndiff = np.sum(test_labels == 1)

        step = 0.001
        max_acc = 0
        best_thr = dmin
        for d in np.arange(dmin, dmax + step, step):
            idx1 = distances <= d
            idx2 = distances > d
            tpr = float(np.sum(np.array(test_labels)[idx1] == 0)) / nsame
            tnr = float(np.sum(np.array(test_labels)[idx2] == 1)) / ndiff
            acc = 0.5 * (tpr + tnr)
            if acc > max_acc:
                max_acc = acc
                best_thr = d

        y_pred = (distances < best_thr).astype(int)
        y_true = (np.array(test_labels) == 0).astype(int)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, -distances)

        cross_results.append([train_name, test_name, max_acc, f1, auc, best_thr])
        print(f"✅ Acc={max_acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | Thr={best_thr:.4f}")

# Save to CSV
df = pd.DataFrame(cross_results, columns=["Train Dataset", "Test Dataset", "Accuracy", "F1 Score", "ROC AUC", "Best Threshold"])
df.to_csv("SigNet_Baseline_CrossDataset.csv", index=False)
print("📄 Cross-dataset results saved to SigNet_Baseline_CrossDataset.csv")

# Plot Heatmap
plt.figure(figsize=(8, 6))
heatmap_data = df.pivot(index="Train Dataset", columns="Test Dataset", values="Accuracy")
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Blues")
plt.title("SigNet Baseline Cross-Dataset Accuracy")
plt.tight_layout()
plt.savefig("SigNet_Baseline_Cross_Heatmap.png")
print("✅ Cross-dataset heatmap saved.")
