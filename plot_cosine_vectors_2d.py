# plot_cosine_vectors_2d.py

from config import BIOMEDICAL_PHRASES, MODEL_NAME
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import math

# Load model
print(f"🔄 Loading model: {MODEL_NAME}...", flush=True)
model = SentenceTransformer(MODEL_NAME)

# Get phrases
phrases = BIOMEDICAL_PHRASES

# Generate embeddings
embeddings = model.encode(phrases)

# Calculate cosine similarities
cos_sim_matrix = cosine_similarity(embeddings)

# Choose the first phrase as anchor (index 0)
anchor_idx = 0
anchor_embedding = embeddings[anchor_idx]

# Calculate angles relative to the anchor
angles = []
for i in range(len(phrases)):
    if i == anchor_idx:
        angles.append(0)  # Anchor is at 0°
    else:
        sim = cos_sim_matrix[anchor_idx, i]
        angle_rad = math.acos(np.clip(sim, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)

# Normalize angles nicely around the right half
angles = np.array(angles)

# Define radius for all vectors (equal length for clarity)
r = 1.0

scale = 0.8 # 60% of the full radial space
x = scale * np.cos(np.radians(angles))
y = scale * np.sin(np.radians(angles))

# Plot
fig, ax = plt.subplots(figsize=(8, 8))

# Draw arrows (smaller, sharper heads)
for i in range(len(phrases)):
    ax.arrow(
        0, 0, x[i], y[i],
        head_width=0.02, head_length=0.04,
        fc='black', ec='black'
    )

    # Draw a dashed arc showing angle between anchor and second phrase
if len(phrases) > 1:
    # Use second phrase (index 1) for example
    theta = np.radians(angles[1])  # angle in radians
    arc = np.linspace(0, theta, 100)  # range from 0 to theta
    arc_radius = 0.3  # smaller radius for visual clarity

    arc_x = arc_radius * np.cos(arc)
    arc_y = arc_radius * np.sin(arc)

    ax.plot(arc_x, arc_y, linestyle='dashed', color='gray', linewidth=1)

    # Annotate angle value
    mid_angle = theta / 2
    ax.text(
        arc_radius * 1.2 * np.cos(mid_angle),
        arc_radius * 1.2 * np.sin(mid_angle),
        f"{round(angles[1])}\u00b0",
        fontsize=9,
        color='gray',
        ha='center',
        va='center'
    )


# Annotate phrases (well farther from tips)
# Annotate phrases (offset and rotated)
for i, phrase in enumerate(phrases):
    label_x = x[i] * 1.35
    label_y = y[i] * 1.35
    ax.text(
        label_x, label_y, phrase,
        fontsize=10,
        ha='center', va='center',
        rotation=angles[i],  # <-- Rotate label along vector angle
        rotation_mode='anchor'  # <-- Keep anchor at center while rotating
    )


# Aesthetic settings
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.2, 1.5)
ax.set_aspect('equal')
plt.title("Cosine Similarity Vectors in 2D Semantic Space")
ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
plt.tight_layout()
plt.show()