from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

# Get x, y, z coordinates
xs = embeddings_3d[:, 0]
ys = embeddings_3d[:, 1]
zs = embeddings_3d[:, 2]

# Plot the points
ax.scatter(xs, ys, zs, s=100, color='blue')

# Annotate points with labels
for i, phrase in enumerate(phrases):
    ax.text(xs[i], ys[i], zs[i], phrase, fontsize=9, ha='center')

    # Draw dotted projection lines to zero planes
    ax.plot([xs[i], xs[i]], [ys[i], ys[i]], [0, zs[i]], linestyle='dotted', color='gray')  # To XY plane
    ax.plot([xs[i], xs[i]], [0, ys[i]], [zs[i], zs[i]], linestyle='dotted', color='gray')  # To XZ plane
    ax.plot([0, xs[i]], [ys[i], ys[i]], [zs[i], zs[i]], linestyle='dotted', color='gray')  # To YZ plane

# Plot light shaded planes at x=0, y=0, z=0
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

# XY plane at Z=0
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.1, color='lightcoral', edgecolor='none')

# XZ plane at Y=0
yy2, zz2 = np.meshgrid(np.linspace(ylim[0], ylim[1], 10), np.linspace(zlim[0], zlim[1], 10))
xx2 = np.zeros_like(yy2)
ax.plot_surface(xx2, yy2, zz2, alpha=0.1, color='grey', edgecolor='none')

# YZ plane at X=0
xx3, zz3 = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(zlim[0], zlim[1], 10))
yy3 = np.zeros_like(xx3)
ax.plot_surface(xx3, yy3, zz3, alpha=0.1, color='green', edgecolor='none')

# Labels and Title
ax.set_title("3D PCA Projection of Biomedical Phrase Embeddings", fontsize=14)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.tight_layout()
plt.show()
