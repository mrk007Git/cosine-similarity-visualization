from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.pyplot as plt
from config import BIOMEDICAL_PHRASES, MODEL_NAME

# Notify user
print("🔄 Loading the language model... (this may take up to a few minutes on first run, depending on your internet speed and machine configuration)\nPlease be patient.\n", flush = True)

# Load model
model = SentenceTransformer(MODEL_NAME)

# Define phrases
phrases = BIOMEDICAL_PHRASES

# Generate embeddings
embeddings = model.encode(phrases, convert_to_tensor=True)

# Compute cosine similarity matrix
cosine_similarities = util.cos_sim(embeddings, embeddings)

# Print formatted results
print("Cosine Similarity Matrix:\n")
for i in range(len(phrases)):
    for j in range(len(phrases)):
        sim_score = cosine_similarities[i][j].item()
        print(f"Similarity({phrases[i]} ↔ {phrases[j]}): {sim_score:.4f}")
    print()  # Blank line between rows

# Convert tensor to numpy array for plotting
cosine_similarities = cosine_similarities.cpu().numpy()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cosine_similarities, 
    annot=True, 
    cmap='coolwarm', 
    xticklabels=phrases, 
    yticklabels=phrases, 
    fmt=".2f",
    cbar=True
)

# Title and styling
plt.title("Cosine Similarity Between Medical Phrases", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()