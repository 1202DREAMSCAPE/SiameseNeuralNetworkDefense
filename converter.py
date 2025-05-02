# weights_convert_to_h5.py

from tensorflow.keras.models import load_model
from SigNet_v1 import create_base_network_signet_dilated as create_base_network_signet

# === Configuration ===
input_shape = (155, 220, 3)
embedding_size = 128

# === Step 1: Create model architecture ===
model = create_base_network_signet(input_shape, embedding_dim=embedding_size)

# === Step 2: Load weights only ===
model.load_weights("CEDAR_base_network_margin0.2.weights.h5")  # change this to your actual file

# === Step 3: Save as complete .h5 model ===
model.save("base_network_complete.h5")
print("✅ Saved full model as base_network_complete.h5")
