from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from config import BIOMEDICAL_PHRASES, MODEL_NAME

# Load model
print("🔄 Loading the language model... (please be patient)", flush=True)
model = SentenceTransformer(MODEL_NAME)

# Define phrases
phrases = BIOMEDICAL_PHRASES

# Generate embeddings
embeddings = model.encode(phrases)

# Apply PCA
pca_full = PCA()
pca_full.fit(embeddings)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_, marker='o')
plt.title('Variance Explained by Each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.xticks(range(1, len(phrases) + 1))  # Only need up to number of samples
plt.grid(True)
plt.tight_layout()
plt.show()

# Then separately apply PCA for actual 3D reduction
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# (Then you can proceed to plot 3D if you want)
