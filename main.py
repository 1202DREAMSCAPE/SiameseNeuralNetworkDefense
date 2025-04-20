import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.optimizers import RMSprop
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
import time

# Define Contrastive Loss
class ContrastiveLoss(Loss):
    def __init__(self, margin=1.0, **kwargs):
        """
        Contrastive loss function for Siamese Network.

        Args:
            margin: The margin value for dissimilar pairs.
        """
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Compute the contrastive loss.

        Args:
            y_true: Ground truth labels (0 for similar, 1 for dissimilar).
            y_pred: Predicted distances between embeddings.

        Returns:
            Loss value.
        """
        y_true = K.cast(y_true, y_pred.dtype)
        positive_loss = (1 - y_true) * K.square(y_pred)
        negative_loss = y_true * K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(positive_loss + negative_loss)

# Dataset Configuration
datasets = {
   "CEDAR": {
        "path": "/Users/christelle/Downloads/EnhancedSiameseNN-Thesis/Dataset/CEDAR",
        "train_writers": list(range(260, 300)),
        "test_writers": list(range(300, 314))
    },
    "BHSig260_Bengali": {
        "path": "/Users/christelle/Downloads/EnhancedSiameseNN-Thesis/Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),
        "test_writers": list(range(71, 101))
    },
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/EnhancedSiameseNN-Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 169)),
        "test_writers": list(range(170, 261))
    }
}

# Function to load data
def load_data(dataset_name, dataset_config):
    generator = SignatureDataGenerator(
        dataset={
            dataset_name: {
                "path": dataset_config["path"],
                "train_writers": dataset_config["train_writers"],
                "test_writers": dataset_config["test_writers"]
            }
        },
        img_height=155,
        img_width=220
    )
    train_data, train_labels = generator.get_train_data()
    test_data, test_labels = generator.get_test_data()

    return train_data, train_labels, test_data, test_labels

# 📊 Plot Training Loss
def plot_loss_curve(history, dataset_name):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training Loss Curve - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{dataset_name}_loss_curve.png")
    plt.close()

# 🔍 CLAHE Visualization
def visualize_clahe(input_img_path, dataset_name):
    image = cv2.imread(input_img_path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    combined = np.hstack((image, enhanced))
    cv2.imwrite(f"{dataset_name}_clahe_comparison.png", combined)

# 🔎 Embedding Space Visualization
def plot_tsne(embeddings, labels, dataset_name):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.title(f"t-SNE Embedding Space - {dataset_name}")
    plt.savefig(f"{dataset_name}_tsne.png")
    plt.close()

# 🕒 Brute-force Distance Timing
def compute_brute_force_time(model, test_data, dataset_name):
    print(f"\n⏱ Measuring Brute-Force Time for {dataset_name}...")
    embeddings = model.predict(test_data)
    start = time.time()
    distances = cdist(embeddings, embeddings, metric='euclidean')
    end = time.time()
    duration = end - start
    print(f"Brute-Force Time on {dataset_name}: {duration:.4f} seconds")
    return duration, embeddings


# Function to visualize class imbalance
def plot_class_imbalance(train_labels, test_labels, dataset_name):
    train_counts = np.bincount(train_labels)
    test_counts = np.bincount(test_labels)

    print(f"Class Imbalance Summary for {dataset_name} Dataset:")
    print(f"Training Data: {train_counts[0]} Genuine, {train_counts[1]} Forged")
    print(f"Testing Data: {test_counts[0]} Genuine, {test_counts[1]} Forged")

    labels = ['Genuine', 'Forged']
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, train_counts, width=0.4, label='Train', color='blue', alpha=0.7)
    plt.bar(x + 0.2, test_counts, width=0.4, label='Test', color='orange', alpha=0.7)
    plt.xticks(x, labels)
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.title(f'Class Imbalance - {dataset_name}')
    plt.legend()
    plt.show()

# Updated function to compute and display metrics
def compute_metrics(y_true, y_pred, dataset_name):
    """
    Computes classification metrics and prints results.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.

    Returns:
        None
    """
    y_pred_labels = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred)

    # Biometric-specific metrics
    gar = recall  # Genuine Acceptance Rate
    frr = 1 - gar  # False Rejection Rate

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (GAR): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Genuine Acceptance Rate (GAR): {gar:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=["Genuine", "Forged"]))

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], "k--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"{dataset_name}_roc_base.png")

# Main Script
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")
    train_data, train_labels, test_data, test_labels = load_data(dataset_name, dataset_config)

    model = create_siamese_network(input_shape=(155, 220, 1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=ContrastiveLoss(margin=1.0))

    print(f"Training on {dataset_name}...")
    start = time.time()
    history = model.fit(train_data, train_labels, epochs=10, batch_size=8, validation_split=0.2, verbose=1)
    end = time.time()

    print(f"Evaluating on {dataset_name}...")
    y_pred = model.predict(test_data)
    compute_metrics(test_labels, y_pred, dataset_name)
    plot_roc_curve(test_labels, y_pred, dataset_name)

    brute_time, embeddings = compute_brute_force_time(model, test_data, dataset_name)
    plot_loss_curve(history, dataset_name)
    plot_tsne(embeddings, test_labels, dataset_name)

        # Optional: Show CLAHE transformation example
    sample_path = os.path.join(dataset_config['path'], "writer_001", "genuine", "01.png")
    if os.path.exists(sample_path):
        visualize_clahe(sample_path, dataset_name)

    print(f"Total Time: {end - start:.2f} seconds | Brute-Force: {brute_time:.4f} sec")

