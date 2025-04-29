import os
import numpy as np
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
    add_noise_to_image,
    apply_smote_per_dataset,
    find_hard_negatives,
    compute_hard_negative_metrics,
    evaluate_threshold
)
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from SigNet_v1 import get_triplet_loss
from SigNet_v1 import create_base_network_signet_dilated as create_base_network_signet
from SigNet_v1 import create_triplet_network
from imblearn.over_sampling import SMOTE
from SignatureDataGenerator import SignatureDataGenerator
from collections import Counter
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

# Embedding Size
EMBEDDING_SIZE = 128  


# --- SOP METRIC FUNCTIONS ---

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
    sample_size = min(1000, len(embeddings))
    if sample_size > 10:
        sample_idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
        silhouette = silhouette_score(embeddings[sample_idx], labels[sample_idx]) if len(np.unique(labels[sample_idx])) > 1 else -1
        intra_class_var = np.mean([np.var(embeddings[labels == i], axis=0).mean() for i in np.unique(labels)])
    else:
        silhouette = -1
        intra_class_var = -1
    return {
        'SOP2_Silhouette': silhouette,
        'SOP2_IntraClass_Var': intra_class_var
    }

def evaluate_sop3(clean_emb, noisy_emb, clean_labels, threshold):
    shifts = np.linalg.norm(clean_emb - noisy_emb, axis=1)
    dists = np.array([np.min(np.linalg.norm(clean_emb[clean_labels == 0] - e, axis=1)) for e in noisy_emb])
    preds = (dists > threshold).astype(int)
    acc_drop = 1 - accuracy_score(clean_labels, preds)
    psnr_list = []
    for c, n in zip(clean_emb, noisy_emb):
        mse = np.mean((c - n) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float('inf')
        psnr_list.append(psnr)
    return {
        'SOP3_Mean_PSNR': np.mean(psnr_list),
        'SOP3_Accuracy_Drop': acc_drop,
        'SOP3_Mean_Shift': np.mean(shifts),
        'SOP3_Max_Shift': np.max(shifts)
    }

# Dataset Configuration
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

# Initialize an empty dictionary to store embeddings for each dataset
dataset_embeddings = {}

# Data Preparation Loop (collect embeddings from all datasets first)
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Preparing data for {dataset_name} ---")

    try:
        train_writers = dataset_config["train_writers"]
        test_writers = dataset_config["test_writers"]

        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=32
        )

        generator.save_dataset_to_csv(f"{dataset_name}_signature_dataset.csv")
        generator.visualize_clahe_effect(output_dir=f"CLAHE_Comparison_{dataset_name}")

        # Debugging: Print train and test writers
        print(f"🔍 Processing train writers: {train_writers}")
        print(f"🔍 Processing test writers: {test_writers}")

        # Create base network
        base_network = create_base_network_signet((155, 220, 3), embedding_dim=EMBEDDING_SIZE)

        # Step 1: Get all raw images and labels (for SMOTE)
        all_images, all_labels = generator.get_all_data_with_labels()

        # Step 2: Convert raw images to embeddings using base_network
        image_embeddings = base_network.predict(all_images)
        print(f"🔎 Class Distribution Before SMOTE for {dataset_name}: {Counter(all_labels)}")

        # Store embeddings in the dictionary for later use
        dataset_embeddings[dataset_name] = (image_embeddings, all_labels)

    except Exception as e:
        print(f"❌ Error preparing dataset {dataset_name}: {e}")
        continue

# Step 3: Apply SMOTE to each dataset independently if imbalance is detected
print("\n🚀 Applying SMOTE to each dataset independently...")
X_final, y_final = apply_smote_per_dataset(dataset_embeddings)

# ✅ Correct: Train on balanced images after per-dataset SMOTE
train_data = tf.data.Dataset.from_tensor_slices((
    (X_final, X_final, X_final),  # Placeholder for triplets
    np.zeros((len(X_final),))  # Dummy labels required for Keras API
)).batch(generator.batch_sz, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Confirm shapes after balancing
print(f"🔎 Shape of X_final: {X_final.shape}")
print(f"🔎 Shape of y_final: {y_final.shape}")

# ============================
# Training Loop (train the model using balanced data)
# ============================
# Training Loop (with Hard Negative Mining)
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Training Model on {dataset_name} (with Hard Negative Mining) ---")

    try:
        # Initialize data generator
        train_writers = dataset_config["train_writers"]
        test_writers = dataset_config["test_writers"]

        generator = SignatureDataGenerator(
            dataset={dataset_name: dataset_config},
            img_height=155,
            img_width=220,
            batch_sz=32
        )

        # ✅ Create base network
        base_network = create_base_network_signet((155, 220, 3), embedding_dim=EMBEDDING_SIZE)

        # ✅ Generate hard-mined triplets (anchor, positive, hardest negative)
        anchor_imgs, positive_imgs, negative_imgs = generator.generate_hard_mined_triplets(base_network)

        # Check shape
        print(f"🔎 Anchor shape: {np.array(anchor_imgs).shape}")
        print(f"🔎 Positive shape: {np.array(positive_imgs).shape}")
        print(f"🔎 Negative shape: {np.array(negative_imgs).shape}")

        # ✅ Create Triplet Network
        triplet_model = create_triplet_network((155, 220, 3), embedding_dim=EMBEDDING_SIZE)

        # ✅ Convert to tf.data.Dataset
        dummy_labels = np.zeros((len(anchor_imgs),))
        train_data = tf.data.Dataset.from_tensor_slices((
            (np.array(anchor_imgs), np.array(positive_imgs), np.array(negative_imgs)),
            dummy_labels
        )).map(lambda x, y: ((x[0], x[1], x[2]), y)) \
         .batch(generator.batch_sz, drop_remainder=True) \
         .prefetch(tf.data.AUTOTUNE)

        # ✅ Compile using get_triplet_loss
        triplet_model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss=get_triplet_loss(margin=0.5)
        )

        steps_per_epoch = max(1, len(anchor_imgs) // generator.batch_sz)
        print(f"🟢 Steps Per Epoch: {steps_per_epoch}, Batch Size: {generator.batch_sz}")

        early_stopping = EarlyStopping(
            monitor='loss',           # or 'val_loss' if using validation
            patience=5,               # Number of epochs to wait before stopping
            restore_best_weights=True,
            verbose=1
        )

        # ✅ Train the model
        history = triplet_model.fit(
            train_data,
            epochs=1,
            steps_per_epoch=steps_per_epoch,
            callbacks=[early_stopping] 
        )

       # ✅ Save the full triplet model architecture (only if needed for retraining)
        triplet_model_json_path = f"{dataset_name}_triplet_model_architecture.json"
        with open(triplet_model_json_path, 'w') as json_file:
            json_file.write(triplet_model.to_json())
        print(f"✅ Full triplet model architecture saved as {triplet_model_json_path}")

        # ✅ Save the full triplet model weights (only if needed for retraining)
        triplet_model_weights_path = f"{dataset_name}_triplet_model.weights.h5"  # must end with `.weights.h5`
        triplet_model.save_weights(triplet_model_weights_path)
        print(f"🚀 Full triplet model weights saved as {triplet_model_weights_path}")

        # ✅ Save the base network architecture (needed for deployment/inference)
        base_network_json_path = f"{dataset_name}_base_network_architecture.json"
        with open(base_network_json_path, 'w') as json_file:
            json_file.write(base_network.to_json())
        print(f"✅ Base network architecture saved as {base_network_json_path}")

        # ✅ Save the base network weights (needed for deployment/inference)
        base_network_weights_path = f"{dataset_name}_base_network.weights.h5"  # must end with `.weights.h5`
        base_network.save_weights(base_network_weights_path)
        print(f"🚀 Base network weights saved as {base_network_weights_path}")

        # Training with Hard Negative Mining completed for dataset
        print(f"Training with Hard Negative Mining completed for {dataset_name}")

    except Exception as e:
        print(f"❌ Error during training with HNM on {dataset_name}: {e}")
        continue

    # ✅ Recompute embeddings from trained model
    all_images, all_labels = generator.get_all_data_with_labels()
    embeddings = base_network.predict(all_images, verbose=0)

    # ============================
    # ✅ Prepare Reference and Query Embeddings (Real-World Evaluation Setup)
    # ============================
    print(f"\n🧪 Real-World Evaluation for {dataset_name}")

    reference_embeddings = []
    reference_labels = []

    query_embeddings = []
    query_labels = []

    # Loop through test writers
    for dataset_path, writer in tqdm(generator.test_writers, desc="🔍 Processing test writers"):
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        genuine_files = sorted([
            os.path.join(genuine_path, f)
            for f in os.listdir(genuine_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        forged_files = sorted([
            os.path.join(forged_path, f)
            for f in os.listdir(forged_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(genuine_files) < 2 or len(forged_files) < 1:
            continue

        # Reference embedding: 1 genuine signature as reference
        reference_img = generator.preprocess_image(genuine_files[0])
        reference_emb = base_network.predict(np.expand_dims(reference_img, axis=0), verbose=0)[0]
        reference_embeddings.append(reference_emb)
        reference_labels.append(writer)

        # Remaining genuine signatures as positive queries
        for img_path in tqdm(genuine_files[1:], leave=False, desc=f"Writer {writer} - Genuine"):
            query_img = generator.preprocess_image(img_path)
            emb = base_network.predict(np.expand_dims(query_img, axis=0), verbose=0)[0]
            query_embeddings.append(emb)
            query_labels.append(("Genuine", writer))

        # Forged signatures as negative queries
        for img_path in tqdm(forged_files, leave=False, desc=f"Writer {writer} - Forged"):
            query_img = generator.preprocess_image(img_path)
            emb = base_network.predict(np.expand_dims(query_img, axis=0), verbose=0)[0]
            query_embeddings.append(emb)
            query_labels.append(("Forged", writer))

    # SOP Metrics Calculation
    genuine_d = [np.min(np.linalg.norm(np.delete(embeddings, i, axis=0) - emb, axis=1))
                   for i, (emb, label) in enumerate(zip(embeddings, all_labels)) if label == 1]
    forged_d = [np.min(np.linalg.norm(np.delete(embeddings, i, axis=0) - emb, axis=1))
                 for i, (emb, label) in enumerate(zip(embeddings, all_labels)) if label == 0]

    sop_metrics_path = f"{dataset_name}_sop_metrics.txt"
    with open(sop_metrics_path, "w") as f:
        f.write("SOP 1 – 🔍 Distance Distributions:\n")
        f.write(f"Genuine mean distance: {np.mean(genuine_d):.4f}\n")
        f.write(f"Forged mean distance:  {np.mean(forged_d):.4f}\n\n")

        # SOP 2 – Time measurement
        start = time.time()
        for emb in embeddings:
            _ = [np.linalg.norm(emb - ref) for ref in embeddings]
        elapsed = time.time() - start
        time_per_query = elapsed / len(embeddings)
        f.write(f"⏱ SOP 2 – Time per query: {time_per_query:.4f}s for {len(embeddings)} samples\n")

        # SOP 3 – Clean vs Noisy Evaluation
        try:
            clean_imgs, clean_lbls = generator.get_unbatched_data()
            noisy_imgs, noisy_lbls = generator.get_unbatched_data(noisy=True)
            clean_emb = base_network.predict(clean_imgs)
            noisy_emb = base_network.predict(noisy_imgs)
            ref_clean = clean_emb[clean_lbls == 0]

            def eval_quality(embs, lbls):
                dists = [np.min(np.linalg.norm(ref_clean - e, axis=1)) for e in embs]
                pred = [1 if d > np.percentile(genuine_d, 90) else 0 for d in dists]
                return accuracy_score(lbls, pred), f1_score(lbls, pred)

            clean_acc, clean_f1 = eval_quality(clean_emb, clean_lbls)
            noisy_acc, noisy_f1 = eval_quality(noisy_emb, noisy_lbls)

            f.write(f"\nSOP 3 – 🧼 Clean Accuracy: {clean_acc:.4f}, F1: {clean_f1:.4f}\n")
            f.write(f"SOP 3 – 🔧 Noisy Accuracy: {noisy_acc:.4f}, F1: {noisy_f1:.4f}\n")
        except Exception as e:
            f.write("⚠️ SOP 3 Evaluation failed: " + str(e) + "\n")

        print(f"📄 SOP metrics saved to {sop_metrics_path}")

    # --- REAL-WORLD EVALUATION STARTS HERE ---

    print(f"\n🧪 Real-World Evaluation for {dataset_name}")

    reference_array = np.array(reference_embeddings)
    reference_norms = reference_array / np.linalg.norm(reference_array, axis=1, keepdims=True)

    # First loop to get distances
    distances = []
    binary_labels = []
    for query, (label_type, _) in zip(query_embeddings, query_labels):
        query_norm = query / np.linalg.norm(query)
        dists = np.linalg.norm(reference_norms - query_norm, axis=1)
        score = np.min(dists)
        distances.append(score)
        binary_labels.append(1 if label_type == "Genuine" else 0)

    # Kernel Density for Optimal Threshold
    genuine_dists = [d for d, l in zip(distances, binary_labels) if l == 1]
    forged_dists = [d for d, l in zip(distances, binary_labels) if l == 0]

    genuine_kde = gaussian_kde(genuine_dists)
    forged_kde = gaussian_kde(forged_dists)
    x_range = np.linspace(min(min(genuine_dists), min(forged_dists)), max(max(genuine_dists), max(forged_dists)), 1000)
    genuine_density = genuine_kde(x_range)
    forged_density = forged_kde(x_range)

    intersection_idx = np.argwhere(np.diff(np.sign(genuine_density - forged_density))).flatten()
    threshold = x_range[intersection_idx][0] if len(intersection_idx) > 0 else np.percentile(genuine_dists, 90)
    print(f"🔑 Optimal Threshold: {threshold:.4f}")

    # Second loop to predict
    y_pred_thresh = [1 if d <= threshold else 0 for d in distances]
    y_true_thresh = binary_labels

    # --- SOP METRICS ---
    sop1 = evaluate_sop1(genuine_dists, forged_dists, binary_labels, distances)
    sop2 = evaluate_sop2(np.array(query_embeddings), np.array(binary_labels))
    try:
        clean_images, clean_labels = generator.get_unbatched_data()
        noisy_images, noisy_labels = generator.get_unbatched_data(noisy=True)
        clean_emb = base_network.predict(clean_images)
        noisy_emb = base_network.predict(noisy_images)
        sop3 = evaluate_sop3(clean_emb, noisy_emb, clean_labels, sop1['SOP1_Threshold'])
    except Exception as e:
        print(f"⚠️ SOP3 Evaluation failed: {e}")
        sop3 = {'SOP3_Mean_PSNR': -1, 'SOP3_Accuracy_Drop': -1, 'SOP3_Mean_Shift': -1, 'SOP3_Max_Shift': -1}

    # Merge all SOP
    enhanced_sop = {**sop1, **sop2, **sop3}

    # Save to CSV
    enhanced_df = pd.DataFrame([enhanced_sop])
    enhanced_csv = f"{dataset_name}_enhanced_SOP_metrics.csv"
    enhanced_df.to_csv(enhanced_csv, index=False)
    print(f"📄 Enhanced SOP metrics saved: {enhanced_csv}")

    # --- Append to evaluation_metrics.txt ---
    eval_path = f"{dataset_name}_evaluation_metrics.txt"
    with open(eval_path, "a") as f:
        f.write("\n\n=== ENHANCED SOP METRICS ===\n")
        for key, value in enhanced_sop.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"✅ SOP metrics appended to {eval_path}")


    # ============================
    # 🔎 Real-World Evaluation – Both Methods
    # ============================

    # Ensure reference embeddings are a normalized matrix
    reference_array = np.array(reference_embeddings)
    reference_norms = reference_array / np.linalg.norm(reference_array, axis=1, keepdims=True)

    # Initialize lists to collect results
    y_true_top1, y_pred_top1, y_scores = [], [], []
    distances = []
    binary_labels = []
    results = []

    # First Loop: Generate Predictions and Scores
    for i, query in enumerate(query_embeddings):
        label = query_labels[i]
        
        # Normalize the query
        query_norm = query / np.linalg.norm(query)
        
        # Compute L2 distances to all references
        dists = np.linalg.norm(reference_norms - query_norm, axis=1)
        
        # Find minimum distance (nearest reference)
        score = np.min(dists)

        # Save the score and actual label
        distances.append(score)
        label_type, _ = query_labels[i]
        binary_labels.append(1 if label_type == "Genuine" else 0)

        # (Optional) Save detailed info for later CSV
        results.append({
            "Query_Index": i,
            "Actual_Label": label_type,
            "Distance": score
        })

    # 📆 Save Raw Distances and Labels to CSV for review
    results_df = pd.DataFrame(results)
    results_csv_path = f"{dataset_name}_raw_distances.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ Raw distances saved as {results_csv_path}")

    # ============================
    # 🔎 Calculate Optimal Threshold Based on Intersection
    # ============================

    # Separate distances
    genuine_dists = [d for d, l in zip(distances, binary_labels) if l == 1]
    forged_dists = [d for d, l in zip(distances, binary_labels) if l == 0]

    # Generate kernel density estimates for both distributions
    genuine_kde = gaussian_kde(genuine_dists)
    forged_kde = gaussian_kde(forged_dists)

    # Create a range of distance values
    x = np.linspace(min(min(genuine_dists), min(forged_dists)), 
                    max(max(genuine_dists), max(forged_dists)), 1000)

    # Calculate the densities
    genuine_density = genuine_kde(x)
    forged_density = forged_kde(x)

    # Find the intersection point
    intersection_idx = np.argwhere(np.diff(np.sign(genuine_density - forged_density))).flatten()
    optimal_threshold = x[intersection_idx][0] if len(intersection_idx) > 0 else np.percentile(genuine_dists, 90)

    print(f"🔑 Optimal Threshold based on Intersection: {optimal_threshold:.4f}")

    # Use the calculated optimal threshold for further predictions
    threshold = optimal_threshold

    # Second Loop: Generate final predictions using the calculated threshold
    for i, query in enumerate(query_embeddings):
        label = query_labels[i]
        query_norm = query / np.linalg.norm(query)
        dists = np.linalg.norm(reference_norms - query_norm, axis=1)
        score = np.min(dists)


        label_type, writer_id = query_labels[i]
        # Predict based on the calculated threshold
        actual_label = label_type
        predicted_label = "Genuine" if score <= threshold else "Forged"

        # Append the result to the list (outside inner loop)
        # Get the dataset name and writer ID
        dataset_name_for_eval = None
        writer_id = label

        # Find which dataset this writer belongs to
        for name, config in datasets.items():
            test_writers = config.get("test_writers", [])

            # Handle standard datasets (list of writer IDs)
            if isinstance(test_writers, list):
                # If it's a list of dicts (like Hybrid), check inside each dict
                if all(isinstance(w, dict) for w in test_writers):
                    writer_ids = [w["writer"] for w in test_writers]
                else:
                    writer_ids = test_writers

                if label in writer_ids:
                    dataset_name_for_eval = name
                    writer_id = label
                    break

        # Append results to the list
        results.append([dataset_name_for_eval, writer_id, actual_label, predicted_label, score])

        # Store metrics for evaluation
        is_genuine = query_labels[i] != -1  # This is already your logic for genuine  # Check the file path to see if it's genuine
        y_true_top1.append(1 if label_type == "Genuine" else 0)
        y_pred_top1.append(1 if predicted_label == "Genuine" else 0)
        y_scores.append(-score)  # Use negative score for ROC

    # Convert the results to a DataFrame after collecting all results
    df_results = pd.DataFrame(results, columns=["Dataset", "Writer ID", "Actual Label", "Predicted Label", "Score"])

    # Filter out forged entries if needed (optional)
    df_results = df_results[df_results['Writer ID'] != -1]

    # Save the DataFrame to a CSV file after the loop
    csv_path = f"{dataset_name}_classification_results.csv"
    df_results.to_csv(csv_path, index=False)

    print(f"✅ CSV file saved: {csv_path}")


    # ============================
    # 📏 Threshold-Based Classification
    # ============================

    # Get only genuine distances
    genuine_distances = [d for d, l in zip(distances, binary_labels) if l == 1]

    # Choose threshold: e.g., 90th percentile of genuine distances
    threshold = optimal_threshold

    # Classify based on threshold
    y_pred_thresh = [1 if d < threshold else 0 for d in distances]
    y_true_thresh = binary_labels

    print("\n📍 Threshold-Based Metrics (Threshold = {:.4f}):".format(threshold))
    print("Accuracy:", accuracy_score(y_true_thresh, y_pred_thresh))
    print("Precision:", precision_score(y_true_thresh, y_pred_thresh))
    print("Recall:", recall_score(y_true_thresh, y_pred_thresh))
    print("F1:", f1_score(y_true_thresh, y_pred_thresh))
    print("ROC AUC (Threshold-Based):", roc_auc_score(y_true_thresh, [-d for d in distances]))
    print("Balanced Accuracy (Threshold-Based):", balanced_accuracy_score(y_true_thresh, y_pred_thresh))

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(np.array(query_embeddings), np.array(binary_labels))
    print(f"Dataset: {dataset_name}, Silhouette Score: {silhouette_avg:.4f}")

    # Save Silhouette Score to a file specific to the dataset
    silhouette_score_path = f"{dataset_name}_silhouette_score.txt"
    with open(silhouette_score_path, "w") as f:
        f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")

    # ============================
    # ⏱ FAISS vs Brute-force Timing Comparison
    # ============================

    print("\n⏱ Timing Comparison:")

    # Brute-force timing
    reference_array = np.array(reference_embeddings)
    start = time.time()
    for query in query_embeddings:
        query_norm = query / np.linalg.norm(query)
        reference_norms = reference_array / np.linalg.norm(reference_array, axis=1, keepdims=True)
        dists = np.linalg.norm(reference_norms - query_norm, axis=1)
        _ = np.argmin(dists)
    brute_time = time.time() - start
    print("Brute-force Total Time:", brute_time)

    num_queries = len(query_embeddings)
    print(f"🕒 Time per query (Brute-force): {brute_time / num_queries:.6f} seconds")


    # ============================
    # 📋 Classification Report
    # ============================
    target_names = ["Forged", "Genuine"]
    print("\n📋 Classification Report:")
    print(classification_report(y_true_top1, y_pred_top1, target_names=target_names))

    # ============================
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true_thresh, [-d for d in distances])
    plt.figure(figsize=(7, 5))
    plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve", color='purple')
    plt.title("Precision vs. Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_precision_recall_curve.png")

    # ============================
    f1s = []
    thresholds_to_check = np.linspace(min(distances), max(distances), 100)
    for t in thresholds_to_check:
        preds = [1 if d < t else 0 for d in distances]
        f1s.append(f1_score(binary_labels, preds))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds_to_check, f1s, label="F1 vs Threshold", color='orange')
    plt.axvline(threshold, color='blue', linestyle='--', label=f"Chosen Threshold: {threshold:.4f}")
    plt.title("F1 Score vs Threshold")
    plt.xlabel("Distance Threshold")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_f1_vs_threshold.png")


    # ============================
    # Separate distances
    # ============================

    genuine_dists = [d for d, l in zip(distances, binary_labels) if l == 1]
    forged_dists = [d for d, l in zip(distances, binary_labels) if l == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(genuine_dists, bins=30, alpha=0.6, label='Genuine Distances', color='green')
    plt.hist(forged_dists, bins=30, alpha=0.6, label='Forged Distances', color='red')
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.title(f'Distance Distribution – {dataset_name}')
    plt.xlabel('L2 Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_distance_distribution.png")  # Save image

    # ============================
    # 🧪 SMOTE Analysis
    # ============================
    print("\n🧪 Running SMOTE on full training embeddings for augmented analysis...")

    # Step 1: Get full training data again
    all_images, all_labels = generator.get_all_data_with_labels()
    image_embeddings = base_network.predict(all_images)

    # Step 2: Check class distribution before applying SMOTE
    label_counts = Counter(all_labels)
    print(f"🔍 Class distribution before SMOTE: {label_counts}")

    min_class = min(label_counts.values())
    max_class = max(label_counts.values())

    if min_class < max_class:
        # Imbalanced, apply SMOTE
        smote = SMOTE(random_state=42)
        smote_embeddings, smote_labels = smote.fit_resample(image_embeddings, all_labels)
        print("✅ SMOTE applied for evaluation.")
        print(f"🔢 Original embeddings: {len(image_embeddings)} | SMOTE-enhanced: {len(smote_embeddings)}")
    else:
        # Balanced — skip SMOTE
        smote_embeddings = image_embeddings
        smote_labels = all_labels
        print("⚠️ Dataset is already balanced — SMOTE not applied.")
        print(f"📦 Embeddings remain the same: {len(smote_embeddings)}")


    # Step 3: Plot t-SNE of Real vs SMOTE embeddings
    # ============================

    def plot_tsne(real_emb, smote_emb, real_labels, smote_labels, title="t-SNE of Real vs SMOTE Embeddings"):
        subset_real = real_emb[:1000]
        subset_fake = smote_emb[-1000:]
        labels = ['Real'] * len(subset_real) + ['SMOTE'] * len(subset_fake)
        combined = np.vstack([subset_real, subset_fake])

        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(combined)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:len(subset_real), 0], reduced[:len(subset_real), 1], alpha=0.5, label='Real', s=10)
        plt.scatter(reduced[len(subset_real):, 0], reduced[len(subset_real):, 1], alpha=0.5, label='SMOTE', s=10)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_tsne_real_vs_smote.png")
        #plt.show()

    plot_tsne(image_embeddings, smote_embeddings, all_labels, smote_labels)
    # ============================
    # 📝 Save Metrics to File
    # ============================
    report = classification_report(y_true_top1, y_pred_top1, target_names=target_names)

    output_path = f"{dataset_name}_evaluation_metrics.txt"
    with open(output_path, "w") as f:
        f.write(f"📍 Dataset: {dataset_name}\n")

        f.write(f"📍 Threshold-Based Metrics (Threshold = {threshold:.4f}):\n")
        f.write(f"Accuracy: {accuracy_score(binary_labels, y_pred_thresh):.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy_score(binary_labels, y_pred_thresh):.4f}\n")
        f.write(f"Precision: {precision_score(binary_labels, y_pred_thresh):.4f}\n")
        f.write(f"Recall: {recall_score(binary_labels, y_pred_thresh):.4f}\n")
        f.write(f"F1: {f1_score(binary_labels, y_pred_thresh):.4f}\n")
        f.write(f"ROC AUC (Threshold-Based): {roc_auc_score(binary_labels, [-d for d in distances]):.4f}\n\n")

        f.write("⏱ Timing Comparison:\n")
        f.write(f"Brute-force Total Time: {brute_time:.4f} seconds\n")
        f.write(f"Time per Query (Brute-force): {brute_time / len(query_embeddings):.6f} seconds\n")
        f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")

        f.write("\n📋 Classification Report:\n")
        f.write(report)

        # ============================
        # 🧪 Clean vs Noisy Evaluation
        # ============================

        print("\n🧪 Evaluating Clean vs Noisy Test Images...")

        # Get clean test images and labels
        # Get noisy test images and labels
        clean_images, clean_labels = generator.get_unbatched_data()
        noisy_images, noisy_labels = generator.get_unbatched_data(noisy=True)
        # Evaluate both

        clean_acc, clean_f1 = evaluate_threshold(base_network, clean_images, clean_labels, threshold)
        noisy_acc, noisy_f1 = evaluate_threshold(base_network, noisy_images, noisy_labels, threshold)

        print(f"✅ Clean Accuracy: {clean_acc:.4f}, F1: {clean_f1:.4f}")
        print(f"⚠ Noisy Accuracy: {noisy_acc:.4f}, F1: {noisy_f1:.4f}")
        print(f"📉 Accuracy Drop: {clean_acc - noisy_acc:.4f}")
        print(f"📉 F1 Drop: {clean_f1 - noisy_f1:.4f}")

        output_noise = f"{dataset_name}_noise_metric.txt"
        # Save to file
        with open(output_noise, "a") as f:
            f.write("\n📉 Robustness to Noise:\n")
            f.write(f"Clean Accuracy: {clean_acc:.4f} | Noisy Accuracy: {noisy_acc:.4f} | Drop: {clean_acc - noisy_acc:.4f}\n")
            f.write(f"Clean F1 Score: {clean_f1:.4f} | Noisy F1 Score: {noisy_f1:.4f} | Drop: {clean_f1 - noisy_f1:.4f}\n")

        # ============================
        # 📊 Hard Negative Mining Evaluation
        # ============================
        print("\n📍 Hard Negative Mining Metrics:")
        hard_negative_indices = find_hard_negatives(distances, binary_labels, threshold, y_pred_thresh)
        hn_ratio, hn_precision, hn_recall = compute_hard_negative_metrics(binary_labels, y_pred_thresh, hard_negative_indices)

        print(f"Hard Negative Ratio: {hn_ratio:.4f}")
        print(f"Hard Negative Precision: {hn_precision:.4f}")
        print(f"Hard Negative Recall: {hn_recall:.4f}")

        output = f"{dataset_name}_hard_negative.txt"
        # 📝 Save Hard Negative Metrics to File
        with open(output, "a") as f:
            f.write("\n📍 Hard Negative Mining Metrics:\n")
            f.write(f"Hard Negative Ratio: {hn_ratio:.4f}\n")
            f.write(f"Hard Negative Precision: {hn_precision:.4f}\n")
            f.write(f"Hard Negative Recall: {hn_recall:.4f}\n")

# Place cross-dataset evaluation block here to run **after** all training and evaluation:
print("\n🧪 Cross-Dataset Evaluation...")
saved_models = {}
for dataset_name in datasets:
    weight_file = f"{dataset_name}.weights.h5"
    if os.path.exists(weight_file):
        saved_models[dataset_name] = weight_file
    else:
        print(f"⚠️ Weight file not found for {dataset_name}: {weight_file}")

print("\n🧪 Starting Cross-Dataset Evaluation...\n")
cross_results = []

for train_name, weight_path in saved_models.items():
    print(f"\n🔁 Using {train_name} model")

    # Load the trained base network
    base_model = create_base_network_signet((155, 220, 3), embedding_dim=EMBEDDING_SIZE)
    base_model.load_weights(weight_path)

    for test_name, config in datasets.items():
        if train_name == test_name:
            continue

        print(f"🔎 Testing on {test_name}")
        generator = SignatureDataGenerator(
            dataset={test_name: config},
            img_height=155,
            img_width=220,
            batch_sz=32
        )

        result = evaluate_threshold(base_model, generator, EMBEDDING_SIZE)

        if result is None:
            print(f"⚠️ Skipping {test_name} due to missing evaluation data.")
            continue

        acc, f1, auc, far, frr, threshold = result
        print(f"✅ {train_name} → {test_name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Threshold={threshold:.4f}")
        cross_results.append([train_name, test_name, acc, f1, auc, threshold])

# Save cross-dataset results
if cross_results:
    df = pd.DataFrame(cross_results, columns=["Train Dataset", "Test Dataset", "Accuracy", "F1 Score", "ROC AUC", "Threshold Used"])
    df.to_csv("cross_dataset_evaluation.csv", index=False)
    print("\n📄 Cross-dataset evaluation saved to cross_dataset_evaluation.csv")

    # Create heatmap of accuracy
    heatmap_data = df.pivot(index="Train Dataset", columns="Test Dataset", values="Accuracy")
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
    plt.title("Cross-Dataset Evaluation – Accuracy Heatmap")
    plt.xlabel("Test Dataset")
    plt.ylabel("Train Dataset")
    plt.tight_layout()
    plt.savefig("cross_dataset_accuracy_heatmap.png")
    print("✅ Heatmap saved as cross_dataset_accuracy_heatmap.png")
else:
    print("\n⚠️ No cross-dataset results generated.")
