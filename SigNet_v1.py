import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda

# ✅ Register the Euclidean distance as a global function
@register_keras_serializable()
def euclidean_distance(vectors):
    """
    Computes the Euclidean distance between two feature vectors.
    """
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.sum(tf.square(x - y), axis=1, keepdims=True), tf.epsilon()))


# ✅ Define Batch-Hard Triplet Loss
def get_triplet_loss(margin=0.3):
    def loss_fn(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, 3, 128])
        y_pred = tf.math.l2_normalize(y_pred, axis=-1)

        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        d_pos = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        loss = tf.maximum(d_pos - d_neg + margin, 0.0)
        return tf.reduce_mean(loss)
    
    return loss_fn

def generate_hard_mined_triplets(self, base_network, batch_size=32):

    anchor_list, positive_list, negative_list = [], [], []

    all_images = []
    all_writer_ids = []

    # Step 1: Load all training images and labels
    for dataset_path, writer in self.train_writers:
        for label_type in ['genuine', 'forged']:
            img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
            if not os.path.exists(img_dir):
                continue

            for img_file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_file)
                img = self.preprocess_image(img_path)

                all_images.append(img)
                all_writer_ids.append(writer)

    all_images_np = np.array(all_images)
    all_embeddings_np = base_network.predict(all_images_np, batch_size=batch_size, verbose=0)

    # Group embeddings by writer
    writer_to_indices = {}
    for idx, writer in enumerate(all_writer_ids):
        writer_to_indices.setdefault(writer, []).append(idx)

    # Step 2: Brute-force mining for hard negatives
    for writer, indices in writer_to_indices.items():
        if len(indices) < 2:
            continue  # Need at least 2 samples for positive pair

        for i in range(len(indices) - 1):
            anchor_idx = indices[i]
            positive_idx = indices[i + 1]

            anchor_emb = all_embeddings_np[anchor_idx]

            # Hardest negative: closest from a different writer
            min_dist = float('inf')
            hardest_neg_idx = None

            for j, other_emb in enumerate(all_embeddings_np):
                if all_writer_ids[j] != writer:  # Ensure it's from a different writer
                    dist = np.linalg.norm(anchor_emb - other_emb)
                    if dist < min_dist:
                        min_dist = dist
                        hardest_neg_idx = j

            if hardest_neg_idx is not None:
                # Create pseudo-filenames for logging
                anchor_name = f"img_{anchor_idx}.png"
                positive_name = f"img_{positive_idx}.png"
                negative_name = f"img_{hardest_neg_idx}.png"

                self.log_triplet(
                    writer=writer,
                    anchor_path=anchor_name,
                    positive_path=positive_name,
                    negative_writer=all_writer_ids[hardest_neg_idx],
                    negative_path=negative_name,
                    negative_label="HardNeg"
                )

                anchor_list.append(all_images[anchor_idx])
                positive_list.append(all_images[positive_idx])
                negative_list.append(all_images[hardest_neg_idx])


    print(f"✅ Total hard-mined triplets: {len(anchor_list)}")
    return anchor_list, positive_list, negative_list

def create_base_network_signet(input_shape, embedding_dim=128):
    # Sequential CNN architecture to generate fixed-size embeddings from signature images
    model = Sequential([
        Input(shape=input_shape),

        # 1st Convolutional Block
        layers.Conv2D(96, (11, 11), activation='relu', strides=(4, 4)),  # Large receptive field, stride for spatial reduction
        layers.BatchNormalization(),  # Normalize activations to stabilize training
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),  # Downsample spatial dimensions

        # 2nd Convolutional Block
        layers.ZeroPadding2D((2, 2)),  # Add padding to maintain dimensions
        layers.Conv2D(256, (5, 5), activation='relu'),  # Deeper features
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Dropout(0.3),  # Regularization

        # 3rd Convolutional Block
        layers.ZeroPadding2D((1, 1)),
        layers.Conv2D(384, (3, 3), activation='relu'),  # More abstract patterns
        layers.ZeroPadding2D((1, 1)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Dropout(0.3),

        # Fully Connected Layers
        layers.Flatten(),  # Flatten features into a vector
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),  # Dense representation
        layers.Dropout(0.5),  # Strong regularization to prevent overfitting

        # Final embedding layer (linear activation for vector space)
        layers.Dense(embedding_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),

        # Normalize embeddings to unit length (L2 norm) to ensure consistent scale
        layers.Lambda(lambda x: K.l2_normalize(x, axis=1), name="l2_normalized")
    ])
    return model

def create_triplet_network_from_existing_base(base_network):
    # Get input shape from base network ((155, 220, 3))
    input_shape = base_network.input_shape[1:]

    # 3 inputs for anchor, positive, and negative images
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    # Pass all three through the same (shared) base network
    encoded_anchor = base_network(input_anchor)
    encoded_positive = base_network(input_positive)
    encoded_negative = base_network(input_negative)

    # Stack the three embeddings into one tensor with shape (batch_size, 3, embedding_dim)
    # This format is useful for computing triplet loss downstream
    stacked = Lambda(lambda x: K.stack(x, axis=1), name="triplet_stack")(
        [encoded_anchor, encoded_positive, encoded_negative]
    )

    # Final model takes 3 inputs and outputs stacked embeddings
    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=stacked)
    return model



def get_contrastive_loss(margin=1.0):
    """
    Contrastive loss with Euclidean distance.
    Positive pairs (label=1) are pulled together,
    Negative pairs (label=0) are pushed apart by at least 'margin'.
    """
    def contrastive_loss(y_true, y_pred):
        """
        y_true: 1 for genuine pair, 0 for forged pair
        y_pred: distance between embeddings
        """
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    return contrastive_loss