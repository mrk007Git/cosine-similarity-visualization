import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define phrases
phrases = [
    "heart attack",
    "myocardial infarction",
    "stroke",
    "cerebrovascular accident"
]

# Generate embeddings
embeddings = model.encode(phrases)

# Calculate cosine similarities
cos_sim_matrix = cosine_similarity(embeddings)

# Choose the first phrase ("heart attack") as the anchor
anchor_idx = 0

# Calculate angles (in degrees) relative to the anchor
angles = []
for i in range(len(phrases)):
    if i == anchor_idx:
        angles.append(0)  # Anchor at 0°
    else:
        sim = cos_sim_matrix[anchor_idx, i]
        angle = math.acos(np.clip(sim, -1.0, 1.0))  # Get angle from cosine
        angles.append(np.degrees(angle))  # Convert to degrees

# Normalize angles around circle
angles = np.array(angles)

# Set radius (constant)
r = 1.0

# Calculate x, y positions
x = r * np.cos(np.radians(angles))
y = r * np.sin(np.radians(angles))

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=200, color='skyblue', edgecolors='black')

# Annotate points
for i, phrase in enumerate(phrases):
    plt.text(x[i], y[i], phrase, ha='center', va='center', fontsize=10)

# Style
plt.title("Cosine Angle Representation of Biomedical Phrases", fontsize=14)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.grid(True)
plt.gca().set_aspect('equal')  # Equal aspect ratio
plt.tight_layout()
plt.show()
