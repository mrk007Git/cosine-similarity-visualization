# Text Embedding Cosine Similarity

This project demonstrates how to compute semantic similarity between biomedical terms using text embeddings and cosine similarity. It also includes visualizations of embeddings in 3D space and an analysis of the explained variance of principal components.

## Features
1. **Cosine Similarity Calculation**:  
   Compute the semantic similarity between phrases using embeddings generated by a pre-trained language model (`all-MiniLM-L6-v2`).
2. **PCA Variance Explained Plot**:  
   Visualize the proportion of variance explained by each principal component after dimensionality reduction.
3. **3D Embedding Visualization**:  
   Reduce high-dimensional embeddings to 3D using PCA and plot them for intuitive human interpretation.
4. **Configurable Phrases and Model**:  
   Easily modify the phrases and model used for embeddings by editing the `config.py` file.

## Requirements
- Python 3.8+
- Required Python packages:
  - `sentence-transformers>=2.2.2`
  - `torch>=1.9.0`
  - `numpy>=1.21.0`
  - `scikit-learn`
  - `seaborn>=0.11.2`
  - `matplotlib>=3.4.3`

Install all required packages with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Calculate Cosine Similarity
Run the following script to compute and display the cosine similarity matrix for predefined biomedical phrases. A heatmap of the similarity matrix will also be displayed.

```bash
python calculate_similarity.py
```

### 2. Plot PCA Variance Explained
Run the following script to visualize the proportion of variance explained by each principal component.

```bash
python pca_variance_explained_plot.py
```

### 3. Visualize Text Embeddings in 3D
Run the following script to reduce embeddings to 3D and visualize them in a 3D scatter plot.

```bash
python plot_text_embeddings_3d.py
```

## Configuration
You can customize the biomedical phrases and the embedding model by editing the `BIOMEDICAL_PHRASES` and `MODEL_NAME` variables in the `config.py` file.

## License
This project is provided for academic demonstration purposes under the MIT License.

## Example Output

### Cosine Similarity 2D Vectors
![Cosine Similarity 2D - heart attack origin](https://github.com/user-attachments/assets/4579c421-39e6-425e-b6ce-469720832ad6)

**Cosine similarity vectors plotted in 2D semantic space, using "heart attack" as the origin. Smaller angles between vectors indicate closer semantic similarity.**

### Cosine Similarity Heatmap
![Cosine Similarity Heatmap](https://github.com/user-attachments/assets/3da8ab7f-4c29-4e43-aa0c-b166e5b69739)

**Heatmap of pairwise cosine similarity scores between biomedical phrases, showing stronger semantic relationships with warmer colors.**

### 3D PCA Projection
![3d pca project of biomedial phrase embeddings](https://github.com/user-attachments/assets/1f41063c-2620-4933-a60e-eb1ca982fa29)

**3D visualization of biomedical phrase embeddings projected onto the first three principal components using PCA, preserving dominant semantic patterns.**

### PCA Variance Explained
![PCA Variance](https://github.com/user-attachments/assets/9f3b2b5d-a053-46bb-baed-d18c12111a92)

**Plot of variance explained by each principal component after dimensionality reduction, highlighting how much semantic information is retained.**

