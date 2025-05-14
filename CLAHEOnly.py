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
import csv
import cv2
import seaborn as sns
from SignatureDataGenerator import SignatureDataGenerator

np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1337)

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
        layers.Dense(128, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    ])
    return model

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

def apply_partial_clahe_per_writer(generator, images, labels, writer_ids, save_dir="clahe_samples2"):
    os.makedirs(save_dir, exist_ok=True)
    processed = []
    sample_log = []

    for wid in np.unique(writer_ids):
        mask = (writer_ids == wid)
        imgs = images[mask]
        lbls = labels[mask]

        genuine = imgs[lbls == 0]
        forged = imgs[lbls == 1]


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

print("🔧 Starting CLAHE preprocessing...")

for dataset_name, config in datasets.items():
    print(f"\n🧪 Preprocessing {dataset_name}")

    try:
        generator = SignatureDataGenerator(
            dataset={dataset_name: config},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE,
        )

        images, class_labels = generator.get_all_data_with_labels()
        _, writer_ids = generator.get_all_data_with_writer_ids()

        X_clahe = apply_partial_clahe_per_writer(
            generator,
            images,
            class_labels,
            writer_ids,
            save_dir=f"clahe_samples2/{dataset_name}"
        )

        os.makedirs("clahe_data", exist_ok=True)
        np.save(f"clahe_data/{dataset_name}_X.npy", X_clahe)
        np.save(f"clahe_data/{dataset_name}_y.npy", class_labels)
        np.save(f"clahe_data/{dataset_name}_writer_ids.npy", writer_ids)

        print(f"✅ CLAHE preprocessing saved for {dataset_name}")
    except Exception as e:
        print(f"❌ Error preprocessing {dataset_name}: {e}")

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Base Model for Dataset {dataset_name}")

    os.makedirs("clahe_data", exist_ok=True)
    images = np.load(f"clahe_data/{dataset_name}_X.npy")
    labels = np.load(f"clahe_data/{dataset_name}_y.npy")

    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    pairs, labels = generator.generate_pairs_from_loaded(images, labels)
    labels = np.array(labels).astype(np.float32)

    #print(f"✅ Loaded {len(pairs)} pairs with labels.")
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

    # early_stopping = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5, #gives out around 50% accuracy
    #     restore_best_weights=True,
    #     verbose=1
    # )
    
    weights_dir = 'base_weights'
    metrics_dir = 'baseline_metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    # Split train pairs
    train_img1 = [pair[0] for pair in train_pairs]
    train_img2 = [pair[1] for pair in train_pairs]

    # Split validation pairs
    val_img1 = [pair[0] for pair in val_pairs]
    val_img2 = [pair[1] for pair in val_pairs]

    # ========== Training ==========
    start_time = time.time()
    history = model.fit(
        [np.array(train_img1), np.array(train_img2)], np.array(train_labels),
        validation_data=([np.array(val_img1), np.array(val_img2)], np.array(val_labels)),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
       # callbacks=[early_stopping],
    )
    train_time = time.time() - start_time
    # ========== Save Weights ==========
    os.makedirs(weights_dir, exist_ok=True)

    # Save final model weights (Siamese full model)
    model.save_weights(os.path.join(weights_dir, f"{dataset_name}_clahe_siamese_model.weights.h5"))

    # Save only the base feature extractor (SigNet)
    base_network.save_weights(os.path.join(weights_dir, f"{dataset_name}_clahe_signet_network.weights.h5"))

    # Save entire model (architecture + weights)
    model.save(f"{dataset_name}_clahe_model.h5")

    print(f"✅ Saved all weights and model for {dataset_name}")

    # ========== Evaluation ==========
    # 1. Get test data
    test_images, test_labels = generator.get_unbatched_data()    
    clean_imgs, clean_labels = test_images, test_labels  # Store clean versions for SOP3
    noisy_imgs, _ = generator.get_unbatched_data(noisy=True)
    noisy_imgs = np.array(noisy_imgs)
    print(f"✅ Noisy images shape: {noisy_imgs.shape}")
    if noisy_imgs.ndim != 4:
        raise ValueError("❌ Noisy images must be a 4D tensor: (batch, height, width, channels)")

    # 2. Extract embeddings
    embeddings = base_network.predict(test_images, verbose=0)
    clean_emb = embeddings  # Store clean embeddings for SOP3
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


    # Diagnostic output
    print("\n=== Embedding Diagnostics ===")
    print(f"Genuine distances - Min: {np.min(genuine_d):.4f}, Max: {np.max(genuine_d):.4f}")
    print(f"Forged distances - Min: {np.min(forged_d):.4f}, Max: {np.max(forged_d):.4f}")

    # Metric evaluation
    sop1_metrics = evaluate_sop1(
        genuine_d=genuine_d,
        forged_d=forged_d,
        binary_labels=binary_labels,
        distances=distances,
        threshold=best_threshold
    )

    # Add extra SOP1 metrics (TPR, TNR, Precision, Recall)
    sop1_metrics = add_extended_sop1_metrics(
        sop1_metrics,
        binary_labels=binary_labels,
        distances=distances,
        threshold=best_threshold
    )

    # Visualization (FAR vs FRR bar chart)
    plot_far_frr_bar_chart(
        roc_far=sop1_metrics['SOP1_FAR'],
        roc_frr=sop1_metrics['SOP1_FRR'],
        dataset_name=dataset_name,
        save_path=f"{dataset_name}_ROC_FAR_FRR_BarChart.png"
    )

    # ========== SOP 2 Evaluation ==========
    sop2_metrics = evaluate_sop2(embeddings, test_labels)

    # ========== SOP 3 Evaluation ==========
    try:
        # Calculate PSNR
        psnr_values = [calculate_psnr(c, n) for c, n in zip(clean_imgs, noisy_imgs)]

        # Evaluate with all required parameters
        sop3_metrics = evaluate_sop3(
            clean_emb=clean_emb,
            noisy_emb=noisy_emb,
            clean_labels=clean_labels,
            threshold=best_threshold
        )
        # Then calculate PSNR separately
        psnr_values = [calculate_psnr(c, n) for c, n in zip(clean_imgs, noisy_imgs)]
        sop3_metrics['SOP3_Mean_PSNR'] = np.mean(psnr_values)
    except Exception as e:
        print(f"⚠️ SOP 3 failed: {e}")
        sop3_metrics = {k: -1 for k in ['SOP3_Mean_PSNR', 'SOP3_Accuracy_Drop', 'SOP3_Mean_Shift', 'SOP3_Max_Shift']}

    # ========== Collect and Save Results ==========
    

    results.append({
        "Dataset": dataset_name,
        "Training_Time": train_time,
        **sop1_metrics,
        **sop2_metrics,
        **sop3_metrics,
        "Accuracy": acc,
        "F1_Score": f1
    })

    pd.DataFrame(results).to_csv("SigNet_Baseline_CLAHE_Results.csv", index=False)
    print(f"✅ Metrics saved for {dataset_name}")

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
    plt.savefig(f"{dataset_name}_baseline_CLAHE_metrics.png")
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
    plt.savefig(f"{dataset_name}_baseline_CLAHE_loss_curve.png")
    plt.close()

print(f"📋 Finished evaluation for {dataset_name}. Current CSV updated.\n")
