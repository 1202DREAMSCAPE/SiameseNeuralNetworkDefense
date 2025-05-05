import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
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
def get_triplet_loss(margin=1.00):
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
    print("🔍 [Brute-force] Generating hard-mined triplets without FAISS...")

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
                anchor_list.append(all_images[anchor_idx])
                positive_list.append(all_images[positive_idx])
                negative_list.append(all_images[hardest_neg_idx])

    print(f"✅ [Brute-force] Total hard-mined triplets: {len(anchor_list)}")
    return anchor_list, positive_list, negative_list



# ✅ Define the SigNet Base Network Architecture with Multi-Scale Feature Learning
def create_base_network_signet_dilated(input_shape, embedding_dim=128):
    """
    Builds the base convolutional neural network with dilated convolutions.
    """
    seq = Sequential()
    
    # First convolutional block
    seq.add(Conv2D(96, (11, 11), activation='relu', strides=(4, 4), input_shape=input_shape))
    seq.add(MaxPooling2D((2, 2), strides=(1, 1)))  # Adjusted pooling
    print(f"Layer Output Shape after First Pooling: {seq.output_shape}")
    
    # Second convolutional block with dilation
    seq.add(Conv2D(256, (5, 5), activation='relu', dilation_rate=2))
    seq.add(MaxPooling2D((2, 2), strides=(1, 1)))  # Adjusted pooling
    seq.add(Dropout(0.3))
    print(f"Layer Output Shape after Second Pooling: {seq.output_shape}")
    
    # Third convolutional block
    seq.add(Conv2D(384, (3, 3), activation='relu'))
    seq.add(Conv2D(256, (3, 3), activation='relu'))
    seq.add(MaxPooling2D((2, 2), strides=(1, 1)))  # Adjusted pooling
    seq.add(Dropout(0.3))
    print(f"Layer Output Shape after Third Pooling: {seq.output_shape}")
    
    # Fully connected layers
    seq.add(Flatten())
    seq.add(Dense(1024, activation='relu'))
    seq.add(Dropout(0.5))
    seq.add(Dense(embedding_dim, activation='linear', name="embedding_layer"))

    # ✅ Normalize the output to unit length
    seq.add(Lambda(lambda x: K.l2_normalize(x, axis=1), name="l2_normalized"))
    return seq

# ✅ Define the Triplet Network (Updated)
def create_triplet_network(input_shape, embedding_dim=128):
    """
    Builds the Triplet Neural Network with the updated base architecture.
    """
    base_network = create_base_network_signet_dilated(input_shape, embedding_dim=embedding_dim)

    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    encoded_anchor = base_network(input_anchor)
    encoded_positive = base_network(input_positive)
    encoded_negative = base_network(input_negative)

    # ✅ Step 1: Apply Lambda with proper output shape
    stacked = Lambda(
        lambda x: K.stack(x, axis=1),
        output_shape=lambda input_shapes: (input_shapes[0][0], 3, input_shapes[0][1]),
        name="triplet_stack"
    )([encoded_anchor, encoded_positive, encoded_negative])

    # ✅ Step 2: Create model from inputs to stacked output
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