from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load model
print("🔄 Loading the language model... (please be patient)", flush=True)
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

# Reduce dimensions from 384D to 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# Plot in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xs = embeddings_3d[:, 0]
ys = embeddings_3d[:, 1]
zs = embeddings_3d[:, 2]

# Plot the points
ax.scatter(xs, ys, zs, s=100, color='blue')

# Annotate each point
for i, phrase in enumerate(phrases):
    ax.text(xs[i], ys[i], zs[i], phrase, fontsize=9, ha='center')

    # Draw dotted lines to planes
    ax.plot([xs[i], xs[i]], [ys[i], ys[i]], [0, zs[i]], linestyle='dotted', color='gray')  # To XY plane
    ax.plot([xs[i], xs[i]], [0, ys[i]], [zs[i], zs[i]], linestyle='dotted', color='gray')  # To XZ plane
    ax.plot([0, xs[i]], [ys[i], ys[i]], [zs[i], zs[i]], linestyle='dotted', color='gray')  # To YZ plane

# Labels and Title
ax.set_title("3D PCA Projection of Biomedical Phrase Embeddings", fontsize=14)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.tight_layout()
plt.show()
