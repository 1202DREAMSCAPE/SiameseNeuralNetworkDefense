import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2

# 🔁 Simplified preprocessing
def preprocess_image_simple(self, img_path):
    if not isinstance(img_path, str) or not os.path.exists(img_path):
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
    try:
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img
    except Exception:
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)

from SignatureDataGenerator import SignatureDataGenerator
SignatureDataGenerator.preprocess_image = preprocess_image_simple

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
    model = Sequential()
    model.add(Input(shape=input_shape))  # ← added here
    model.add(layers.Conv2D(96, (11,11), activation='relu', strides=(4,4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(layers.ZeroPadding2D((2,2)))
    model.add(layers.Conv2D(256, (5,5), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Conv2D(384, (3,3), activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Conv2D(256, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    return model


IMG_SHAPE = (155, 220, 3)
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 1.0
results = []

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

for dataset_name, config in datasets.items():
    print(f"\n\U0001f4e6 Processing {dataset_name}")

    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    pairs, labels = generator.generate_pairs()
    pairs, labels = np.array(pairs), np.array(labels).astype(np.float32)
    val_split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:val_split], pairs[val_split:]
    train_labels, val_labels = labels[:val_split], labels[val_split:]

    input_a = Input(shape=IMG_SHAPE)
    input_b = Input(shape=IMG_SHAPE)
    base_network = create_base_network_signet(IMG_SHAPE)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    optimizer = RMSprop(learning_rate=0.0001, clipnorm=1.0)
    model.compile(loss=get_contrastive_loss(margin=MARGIN), optimizer=optimizer)

    model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
        batch_size=BATCH_SIZE, epochs=EPOCHS
    )

    model.save(f"{dataset_name}_signet_model.keras")

    test_images, test_labels = generator.get_unbatched_data()
    embeddings = base_network.predict(test_images, verbose=0)

    # === SOP 2: All-vs-All comparison (realistic)
    reference_embeddings = embeddings.copy()
    query_embeddings = embeddings.copy()

    distances_slow = []
    binary_labels_slow = []
    start = time.time()

    for i, emb in enumerate(query_embeddings):
        label = test_labels[i]
        dist_to_all = [np.linalg.norm(emb - ref) for j, ref in enumerate(reference_embeddings) if i != j]
        distances_slow.append(min(dist_to_all))
        binary_labels_slow.append(label)

    elapsed_slow = time.time() - start
    time_per_query_slow = elapsed_slow / len(query_embeddings)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(binary_labels_slow, distances_slow, pos_label=1)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    preds = [1 if d > best_threshold else 0 for d in distances_slow]

    # === SOP 1: Distance Analysis
    genuine_d = []
    for i, (emb, label) in enumerate(zip(query_embeddings, test_labels)):
        if label == 0:
            dists = [np.linalg.norm(emb - ref) for j, ref in enumerate(reference_embeddings) if i != j]
            if dists:
                genuine_d.append(min(dists))
    forged_d = [d for d, l in zip(distances_slow, binary_labels_slow) if l == 1]

    print(f"\nSOP 1 – 🔍 Distance Distributions:")
    print(f"Genuine mean distance: {np.mean(genuine_d):.4f}")
    print(f"Forged mean distance:  {np.mean(forged_d):.4f}")
    plt.hist(genuine_d, bins=30, alpha=0.6, label='Genuine')
    plt.hist(forged_d, bins=30, alpha=0.6, label='Forged')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold: {best_threshold:.2f}')
    plt.title("Distance Distribution (SOP 1: Imbalance)")
    plt.legend()
    plt.savefig(f"{dataset_name}_sop1_imbalance_distance_hist.png")
    plt.close()

    print(f"\n⏱ SOP 2 – Time per query: {time_per_query_slow:.4f}s for {len(query_embeddings)} samples")
    plt.figure()
    plt.bar(["SOP 2 – Slow Total Time"], [elapsed_slow])
    plt.title("SOP 2 – Simulated Scalability Load")
    plt.ylabel("Seconds")
    plt.savefig(f"{dataset_name}_sop2_heavy_scalability.png")
    plt.close()

    # === SOP 3: Clean vs Noisy Evaluation
    try:
        clean_imgs, clean_lbls = generator.get_unbatched_data()
        noisy_imgs, noisy_lbls = generator.get_unbatched_data(noisy=True)
        clean_emb = base_network.predict(clean_imgs)
        noisy_emb = base_network.predict(noisy_imgs)
        ref_clean = clean_emb[clean_lbls == 0]

        def eval_quality(embs, lbls):
            dists = [np.min(np.linalg.norm(ref_clean - e, axis=1)) for e in embs]
            pred = [1 if d > best_threshold else 0 for d in dists]
            return accuracy_score(lbls, pred), f1_score(lbls, pred)

        clean_acc, clean_f1 = eval_quality(clean_emb, clean_lbls)
        noisy_acc, noisy_f1 = eval_quality(noisy_emb, noisy_lbls)

        print(f"\nSOP 3 – 🧼 Clean Accuracy: {clean_acc:.4f}, F1: {clean_f1:.4f}")
        print(f"SOP 3 – 🔧 Noisy Accuracy: {noisy_acc:.4f}, F1: {noisy_f1:.4f}")
    except Exception as e:
        print("⚠️ SOP 3 Evaluation failed:", e)
        clean_acc = clean_f1 = noisy_acc = noisy_f1 = -1

    # Final Evaluation
    acc = accuracy_score(binary_labels_slow, preds)
    f1 = f1_score(binary_labels_slow, preds)
    auc = roc_auc_score(binary_labels_slow, [-d for d in distances_slow])

    results.append({
        "Dataset": dataset_name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Threshold": best_threshold,
        "SOP 2 – Time per Query": time_per_query_slow,
        "SOP 3 – Clean Accuracy": clean_acc,
        "SOP 3 – Clean F1": clean_f1,
        "SOP 3 – Noisy Accuracy": noisy_acc,
        "SOP 3 – Noisy F1": noisy_f1,
    })

# ✅ Save results
pd.DataFrame(results).to_csv("SigNet_SOP_Evaluation.csv", index=False)
print("\n✅ Results saved to SigNet_SOP_Evaluation.csv")
