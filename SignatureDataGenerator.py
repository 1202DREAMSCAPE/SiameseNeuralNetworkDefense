import numpy as np
import os
import cv2
import random
import tensorflow as tf
import pandas as pd
import skimage
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from utils import (
    add_noise_to_image
)
#import faiss

# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)

class SignatureDataGenerator:
    def __init__(self, dataset, img_height=155, img_width=220, batch_sz=8):
        self.dataset = dataset
        self.dataset_name = list(dataset.keys())[0]
        self.img_height = img_height
        self.img_width = img_width
        self.batch_sz = batch_sz
        self.train_writers = []
        self.test_writers = []
        self._load_writers()
        

    def _load_writers(self):
        """Load writer directories and validate existence."""
        for dataset_name, dataset_info in self.dataset.items():
            dataset_path = dataset_info["path"]
            train_writers = dataset_info["train_writers"]
            test_writers = dataset_info["test_writers"]

            for writer in train_writers + test_writers:
                if isinstance(writer, dict):
                    writer_path = os.path.join(writer["path"], f"writer_{writer['writer']:03d}")
                else:
                    writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")

                if os.path.exists(writer_path):
                    if writer in train_writers:
                        self.train_writers.append((dataset_path, writer))
                    else:
                        self.test_writers.append((dataset_path, writer))
                else:
                    print(f"⚠ Warning: Writer directory not found: {writer_path}")

    def get_triplet_data(self, writers_list):
        """Generate triplet data formatted for TensorFlow's Dataset API."""
        triplets = []

        for dataset_path, writer in writers_list:
            if (dataset_path, writer) not in self.train_writers:
                print(f"⚠ Skipping writer {writer} (not in train_writers)")
                continue  

            writer_triplets = self.generate_triplets(dataset_path, writer)
            if writer_triplets:
                triplets.extend(writer_triplets)
            
            # Debug: Print how many triplets were generated per writer
            if writer_triplets:
                print(f"🟢 Writer {writer} generated {len(writer_triplets)} triplets.")
            else:
                print(f"⚠ Writer {writer} has no valid triplets.")

        def generator():
            for anchor, positive, negative in triplets:
                yield (anchor, positive, negative)  # No labels, only images

        output_signature = (
            (
                tf.TensorSpec(shape=(self.img_height, self.img_width, 3), dtype=tf.float32),  # Anchor
                tf.TensorSpec(shape=(self.img_height, self.img_width, 3), dtype=tf.float32),  # Positive
                tf.TensorSpec(shape=(self.img_height, self.img_width, 3), dtype=tf.float32)   # Negative
            )
        )

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(self.batch_sz)

    def preprocess_image(self, img_path):
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"⚠ Warning: Missing image file: {img_path if isinstance(img_path, str) else 'Invalid Path Type'}")
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠ Warning: Unable to read image {img_path}")
                return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height))

            # Normalize to [-1, 1]
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5

            return img

        except Exception as e:
            print(f"⚠ Error processing image {img_path}: {e}")
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)

    def get_all_data_with_labels(self):
        """
        Collect all images and their labels (0 = genuine, 1 = forged).
        """
        images = []
        labels = []
        for dataset_path, writer in self.train_writers:
            genuine_path = os.path.join(dataset_path, f"writer_{writer:03d}", "genuine")
            forged_path = os.path.join(dataset_path, f"writer_{writer:03d}", "forged")

            # Collect genuine images (label 0)
            if os.path.exists(genuine_path):
                for img_file in os.listdir(genuine_path):
                    img_path = os.path.join(genuine_path, img_file)
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(0)
            else:
                print(f"⚠ Warning: Missing genuine folder for writer {writer}")

            # Collect forged images (label 1)
            if os.path.exists(forged_path):
                for img_file in os.listdir(forged_path):
                    img_path = os.path.join(forged_path, img_file)
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(1)
            else:
                print(f"⚠ Warning: Missing forged folder for writer {writer}")

        return np.array(images), np.array(labels)

    def get_all_data_with_writer_ids(self):
        """
        Return CLAHE‑preprocessed images + WRITER‑ID labels
        (label = writer id, not 0/1).
        """
        images, writer_ids = [], []
        for dataset_path, writer in self.train_writers:
            for label_type in ["genuine", "forged"]:
                img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
                if not os.path.exists(img_dir):
                    continue
                for fn in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, fn)
                    images.append(self.preprocess_image(img_path))
                    writer_ids.append(writer)
        return np.array(images), np.array(writer_ids)

    def log_triplet(self, writer, anchor_path, positive_path, negative_writer, negative_path, negative_label):
        """Log the generated triplet to a CSV file for monitoring."""
        csv_path = f"triplet_monitoring_{self.dataset_name}.csv"
        triplet_info = {
            "Anchor": f"Writer_{writer}_Genuine_{os.path.basename(anchor_path)}",
            "Positive": f"Writer_{writer}_Genuine_{os.path.basename(positive_path)}",
            "Negative": f"Writer_{negative_writer}_{negative_label}_{os.path.basename(negative_path)}",
            "Negative Label": negative_label,
        }
        
        df = pd.DataFrame([triplet_info])
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)

    def generate_triplets(self, dataset_path, writer):
        """Generate writer-independent triplets."""
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        # Check if paths exist
        if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
            print(f"⚠ Warning: Missing images for writer {writer}")
            return []

        # Load genuine and forged files for the current writer
        genuine_files = sorted([os.path.join(genuine_path, f) for f in os.listdir(genuine_path) 
                                if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        forged_files = sorted([os.path.join(forged_path, f) for f in os.listdir(forged_path) 
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        # Ensure sufficient images exist
        if len(genuine_files) < 2 or len(forged_files) < 1:
            print(f"⚠ Not enough images for writer {writer} (genuine: {len(genuine_files)}, forged: {len(forged_files)})")
            return []

        triplets = []

        # Select other writers for the negative sample
        other_writers = [w for d, w in self.train_writers + self.test_writers if w != writer]
        
        for i in range(len(genuine_files) - 1):
            try:
                anchor = self.preprocess_image(genuine_files[i])
                positive = self.preprocess_image(genuine_files[i + 1])

                negative_writer = random.choice(other_writers)
                negative_writer_path = os.path.join(dataset_path, f"writer_{negative_writer:03d}")

                # Randomly choose between genuine or forged negative
                if random.random() > 0.5:
                    negative_label = "Genuine"
                    negative_files = sorted([os.path.join(negative_writer_path, "genuine", f) 
                                            for f in os.listdir(os.path.join(negative_writer_path, "genuine")) 
                                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                else:
                    negative_label = "Forged"
                    negative_files = sorted([os.path.join(negative_writer_path, "forged", f) 
                                            for f in os.listdir(os.path.join(negative_writer_path, "forged")) 
                                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])

                if not negative_files:
                    continue

                negative = self.preprocess_image(random.choice(negative_files))

                if anchor.shape != (self.img_height, self.img_width, 3) or \
                positive.shape != (self.img_height, self.img_width, 3) or \
                negative.shape != (self.img_height, self.img_width, 3):
                    continue

                # Log the triplet with the correct negative label
                self.log_triplet(writer, genuine_files[i], genuine_files[i + 1], negative_writer, random.choice(negative_files), negative_label)
                triplets.append((anchor, positive, negative))

            except Exception as e:
                print(f"⚠ Error processing triplet for writer {writer}: {e}")
        return triplets

    def get_train_data(self):
        """Generate triplet training data using TensorFlow Dataset."""
        return self.get_triplet_data(self.train_writers).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    def get_test_data(self):
        """Fetch test data WITHOUT generating triplets (to keep it untouched)."""
        test_images = []
        test_labels = []

        for dataset_path, writer in self.test_writers:
            writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
            genuine_path = os.path.join(writer_path, "genuine")
            forged_path = os.path.join(writer_path, "forged")

            if os.path.exists(genuine_path):
                for img_file in os.listdir(genuine_path):
                    img_path = os.path.join(genuine_path, img_file)
                    test_images.append(self.preprocess_image(img_path))
                    test_labels.append(0)

            if os.path.exists(forged_path):
                for img_file in os.listdir(forged_path):
                    img_path = os.path.join(forged_path, img_file)
                    test_images.append(self.preprocess_image(img_path))
                    test_labels.append(1)

        return tf.data.Dataset.from_tensor_slices((np.array(test_images), np.array(test_labels))).batch(self.batch_sz)

    def save_dataset_to_csv(self, output_path="signature_dataset.csv"):
        """
        Save the dataset information to a CSV file.
        """
        data_entries = []

        # Iterate through train and test writers
        for dataset_path, writer in self.train_writers + self.test_writers:
            genuine_path = os.path.join(dataset_path, f"writer_{writer:03d}", "genuine")
            forged_path = os.path.join(dataset_path, f"writer_{writer:03d}", "forged")

            # Collect genuine images
            if os.path.exists(genuine_path):
                for img_file in os.listdir(genuine_path):
                    img_path = os.path.join(genuine_path, img_file)
                    data_entries.append([os.path.basename(dataset_path), writer, img_path, "Genuine"])

            # Collect forged images
            if os.path.exists(forged_path):
                for img_file in os.listdir(forged_path):
                    img_path = os.path.join(forged_path, img_file)
                    data_entries.append([os.path.basename(dataset_path), writer, img_path, "Forged"])

        # Convert to DataFrame
        df = pd.DataFrame(data_entries, columns=["Dataset", "Writer ID", "Image Path", "Label"])

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset information saved to: {output_path}")

    def get_noisy_test_data(self):
        """Fetch test data with added noise."""
        test_images = []
        test_labels = []

        for dataset_path, writer in self.test_writers:
            writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
            genuine_path = os.path.join(writer_path, "genuine")
            forged_path = os.path.join(writer_path, "forged")

            if os.path.exists(genuine_path):
                for img_file in os.listdir(genuine_path):
                    img_path = os.path.join(genuine_path, img_file)
                    img = self.preprocess_image(img_path)
                    noisy_img = add_noise_to_image(img)
                    test_images.append(noisy_img)
                    test_labels.append(0)

            if os.path.exists(forged_path):
                for img_file in os.listdir(forged_path):
                    img_path = os.path.join(forged_path, img_file)
                    img = self.preprocess_image(img_path)
                    noisy_img = add_noise_to_image(img)
                    test_images.append(noisy_img)
                    test_labels.append(1)

        return tf.data.Dataset.from_tensor_slices((np.array(test_images), np.array(test_labels))).batch(self.batch_sz)

    def get_unbatched_data(self, noisy=False):
        dataset = self.get_noisy_test_data() if noisy else self.get_test_data()
        images, labels = [], []
        for img, label in dataset.unbatch():
            images.append(img.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    def visualize_clahe_effect(self, output_dir="CLAHE_Comparison"):
        os.makedirs(output_dir, exist_ok=True)

        image_complexities = []

        for dataset_path, writer in self.train_writers:
            if isinstance(writer, dict):
                writer_id = writer["writer"]
                dataset_path = writer["path"]
            else:
                writer_id = writer

            genuine_path = os.path.join(dataset_path, f"writer_{writer_id:03d}", "genuine")
            if not os.path.exists(genuine_path):
                print(f"⚠ Skipping missing genuine path: {genuine_path}")
                continue

            for img_file in os.listdir(genuine_path):
                img_path = os.path.join(genuine_path, img_file)

                try:
                    raw = cv2.imread(img_path)
                    if raw is None:
                        continue

                    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                    entropy = shannon_entropy(gray)

                    image_complexities.append((entropy, img_path, dataset_path, writer_id))

                except Exception as e:
                    print(f"⚠ Error reading image {img_path}: {e}")

        # Sort by complexity descending (highest entropy first)
        top_complex = sorted(image_complexities, key=lambda x: x[0], reverse=True)[:5]

        for i, (entropy, img_path, dataset_path, writer_id) in enumerate(top_complex):
            raw = cv2.imread(img_path)
            raw_resized = cv2.resize(raw, (self.img_width, self.img_height))

            # Apply CLAHE
            lab = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Display side-by-side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            axs[1].set_title("After CLAHE")
            axs[1].axis("off")

            fig.suptitle(f"Writer {writer_id} | Entropy: {entropy:.2f}", fontsize=12)
            fig.tight_layout()

            # Save the side-by-side image
            save_name = f"{os.path.basename(dataset_path)}_writer{writer_id:03d}_comparison_{i+1}.jpg"
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path)
            plt.close()

            print(f"🖼 Saved CLAHE comparison to: {save_path}")

    def generate_hard_mined_triplets(self, base_network, batch_size=32):
        print("🔍 Generating hard-mined triplets ...")

        # Triplet Loss is implemented following these steps:
        # 1. For each writer in the training set:
        anchor_list, positive_list, negative_list = [], [], []
        all_images = []
        all_image_paths = []
        all_writer_ids = []

        for dataset_path, writer in self.train_writers:
            if os.path.basename(dataset_path) != self.dataset_name:
                continue

            # 1.1. Collect all signature images (genuine and forged)
            for label_type in ['genuine', 'forged']:
                img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
                if not os.path.exists(img_dir): continue

                for img_file in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_file)
                    img = self.preprocess_image(img_path)
                    all_images.append(img)
                    all_image_paths.append(img_path) 
                    all_writer_ids.append(writer)

        # 1.2. Compute embeddings using the current base model
        all_images_np = np.array(all_images)
        all_embeddings_np = base_network.predict(all_images_np, batch_size=batch_size, verbose=0)

        # Create mapping: writer → list of image indices
        writer_to_indices = {}
        for idx, writer in enumerate(all_writer_ids):
            writer_to_indices.setdefault(writer, []).append(idx)

        # 2. For each pair of genuine images from the same writer:
        for writer, indices in writer_to_indices.items():
            if len(indices) < 2:
                continue

            for i in range(len(indices) - 1):
                # 2.1. Set anchor and positive as consecutive genuine images
                anchor_idx = indices[i]
                positive_idx = indices[i + 1]
                anchor_emb = all_embeddings_np[anchor_idx]

                # 2.2. Search the entire dataset for the hardest negative
                min_dist = float('inf')
                hardest_neg_idx = None

                for j, other_emb in enumerate(all_embeddings_np):
                    if all_writer_ids[j] != writer:  # 2.2.1. Only consider different writers
                        dist = np.linalg.norm(anchor_emb - other_emb)  # Compute L2 distance
                        if dist < min_dist:
                            min_dist = dist
                            hardest_neg_idx = j  # 2.2.2. Keep the one with the smallest distance

                if hardest_neg_idx is not None:
                    # 3. Form a triplet: (anchor, positive, hardest negative)
                    anchor_list.append(all_images[anchor_idx])
                    positive_list.append(all_images[positive_idx])
                    negative_list.append(all_images[hardest_neg_idx])

                    # 📝 4. Log the triplet for traceability
                    anchor_path = all_image_paths[anchor_idx]
                    positive_path = all_image_paths[positive_idx]
                    negative_path = all_image_paths[hardest_neg_idx]
                    negative_writer = all_writer_ids[hardest_neg_idx]

                    # Determine label type for logging
                    if "genuine" in negative_path.lower():
                        neg_label = "Genuine"
                    elif "forged" in negative_path.lower():
                        neg_label = "Forged"
                    else:
                        neg_label = "Unknown"

                    self.log_triplet(writer, anchor_path, positive_path, negative_writer, negative_path, neg_label)

        print(f"✅ Total hard-mined triplets: {len(anchor_list)}")
        return anchor_list, positive_list, negative_list


    def generate_pairs(self):
        """
        Generate positive and negative pairs for contrastive loss training.
        Returns:
            pairs: list of (img1, img2)
            labels: list of 1 (genuine pair) or 0 (forged/different writer pair)
        """
        import random

        all_images, all_labels = self.get_all_data_with_labels()
        label_to_images = {}

        for img, label in zip(all_images, all_labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img)

        pairs = []
        labels = []

        # Create positive pairs (same label)
        for label in label_to_images:
            images = label_to_images[label]
            if len(images) > 1:
                for i in range(len(images) - 1):
                    pairs.append((images[i], images[i + 1]))
                    labels.append(1)

        # Create negative pairs (different labels)
        all_labels_set = list(label_to_images.keys())
        for _ in range(len(pairs)):  # match the number of positive pairs
            label1, label2 = random.sample(all_labels_set, 2)
            img1 = random.choice(label_to_images[label1])
            img2 = random.choice(label_to_images[label2])
            pairs.append((img1, img2))
            labels.append(0)

        return pairs, labels

    def preprocess_image_from_array(self, img_array):
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        to the L-channel of an RGB image passed as a NumPy array.
        Normalizes the output to [-1, 1].
        """
        try:
            # Check if the image is normalized (float [0,1]) and convert to uint8 [0,255] if needed
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)

            # Resize image to match input size requirements
            img = cv2.resize(img_array, (self.img_width, self.img_height))

            # convert RGB image to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

            # split LAB into separate channels
            l, a, b = cv2.split(lab)

            # Apply CLAHE to the L (lightness) channel to locally enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Merge the enhanced L channel back with original A and B, convert back to RGB
            merged = cv2.merge((cl, a, b))
            img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

            # Normalize pixel values from [0, 255] to [-1, 1] for model input
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5

            return img

        except Exception as e:
            # Handle errors gracefully and return a black image of expected shape
            print(f"⚠ Error in preprocess_image_from_array: {e}")
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)


    def generate_pairs_from_loaded(self, images, labels):
        """
        Generate contrastive pairs (positive/negative) from preloaded images and labels.
        Assumes label 0 = genuine, label 1 = forged.
        """
        import random

        label_to_images = {}
        for img, lbl in zip(images, labels):
            if lbl not in label_to_images:
                label_to_images[lbl] = []
            label_to_images[lbl].append(img)

        pairs = []
        pair_labels = []

        # Positive pairs (same class)
        for label in label_to_images:
            imgs = label_to_images[label]
            if len(imgs) > 1:
                for i in range(len(imgs) - 1):
                    pairs.append((imgs[i], imgs[i + 1]))
                    pair_labels.append(1)

        # Negative pairs (different classes)
        label_keys = list(label_to_images.keys())
        for _ in range(len(pairs)):
            l1, l2 = random.sample(label_keys, 2)
            img1 = random.choice(label_to_images[l1])
            img2 = random.choice(label_to_images[l2])
            pairs.append((img1, img2))
            pair_labels.append(0)

        return pairs, pair_labels
