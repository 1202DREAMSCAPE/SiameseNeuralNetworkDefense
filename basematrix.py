import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from SignatureDataGenerator import SignatureDataGenerator

# ========== CONFIG ==========
IMG_SHAPE = (155, 220, 3)
BATCH_SIZE = 32
weights_dir = "base_weights"
results_dir = "cross_dataset_results"
os.makedirs(results_dir, exist_ok=True)

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

def compute_verification_metrics(embeddings, labels):
    genuine_embs = embeddings[labels == 0]
    ref = np.mean(genuine_embs, axis=0)
    distances = np.linalg.norm(embeddings - ref, axis=1)
    y_true = 1 - labels.astype(int)  # 1 = genuine, 0 = forged for ROC

    fpr, tpr, thresholds = roc_curve(y_true, distances)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except:
        eer = -1

    optimal_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[optimal_idx]
    y_pred = (distances < best_thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        far = FP / (FP + TN) if (FP + TN) != 0 else np.nan
        frr = FN / (FN + TP) if (FN + TP) != 0 else np.nan
    else:
        far = frr = np.nan

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, distances),
        'threshold': best_thr,
        'eer': eer,
        'far': far,
        'frr': frr,
        'distances': distances,
        'predictions': y_pred
    }

def plot_distance_distributions(genuine_d, forged_d, threshold, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(genuine_d, label='Genuine', fill=True)
    sns.kdeplot(forged_d, label='Forged', fill=True)
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title(f"{dataset_name} Distance Distribution")
    plt.xlabel("Embedding Distance")
    plt.legend()
    plt.savefig(f"{results_dir}/{dataset_name}_distances.png")
    plt.close()

# ========== Main Evaluation ==========
cross_results = []

for train_name in datasets.keys():
    model = create_base_network_signet(IMG_SHAPE)
    weight_path = os.path.join(weights_dir, f"{train_name}_signet_network.weights.h5")

    if not os.path.exists(weight_path):
        print(f"❌ Weights not found for {train_name}, skipping.")
        continue

    model.load_weights(weight_path)
    print(f"\n🔁 Evaluating model trained on {train_name}")

    for test_name, config in datasets.items():
        print(f"🔍 Testing on {test_name}")
        generator = SignatureDataGenerator(
            dataset={test_name: config},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE
        )
        test_images, test_labels = generator.get_unbatched_data()
        embeddings = model.predict(test_images, verbose=0)

        metrics = compute_verification_metrics(embeddings, test_labels)
        distances = metrics['distances']
        predictions = metrics['predictions']

        # Save per-sample predictions
        results_per_sample = [
            {
                "Train Dataset": train_name,
                "Test Dataset": test_name,
                "True Label": int(test_labels[i]),
                "Predicted Label": int(predictions[i]),
                "Distance": float(distances[i])
            }
            for i in range(len(test_labels))
        ]
        csv_name = os.path.join(results_dir, f"{train_name}_to_{test_name}_sample_predictions.csv")
        pd.DataFrame(results_per_sample).to_csv(csv_name, index=False)
        print(f"📄 Saved detailed predictions to: {csv_name}")

        ref = np.mean(embeddings[test_labels == 0], axis=0)
        genuine_d = [np.linalg.norm(e - ref) for i, e in enumerate(embeddings) if test_labels[i] == 0]
        forged_d = [np.linalg.norm(e - ref) for i, e in enumerate(embeddings) if test_labels[i] == 1]

        cross_results.append([
            train_name,
            test_name,
            metrics['accuracy'],
            metrics['f1'],
            metrics['auc'],
            metrics['eer'],
            metrics['threshold'],
            metrics['far'],
            metrics['frr']
        ])

        print(f"✅ {train_name}→{test_name} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")

        plot_distance_distributions(genuine_d, forged_d, metrics['threshold'], f"{train_name}_to_{test_name}")

# Save overall results
df = pd.DataFrame(
    cross_results,
    columns=["Train Dataset", "Test Dataset", "Accuracy", "F1 Score", "ROC AUC", "EER", "Threshold", "FAR", "FRR"]
)
df.to_csv(os.path.join(results_dir, "cross_dataset_results.csv"), index=False)

# Accuracy heatmap
plt.figure(figsize=(10, 8))
heatmap_data = df.pivot(index="Train Dataset", columns="Test Dataset", values="Accuracy")
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Greens")
plt.title("Cross-Dataset Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_heatmap.png"))
plt.close()

print("\n✅ Evaluation complete. Results saved to:", results_dir)