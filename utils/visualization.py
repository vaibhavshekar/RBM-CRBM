import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_tsne(real, synthetic, labels_real=None, labels_synth=None, title="t-SNE Visualization"):
    real = real.numpy()
    synthetic = synthetic.numpy()
    data_combined = np.vstack((real, synthetic))
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(data_combined)

    plt.figure(figsize=(10,6))
    plt.scatter(data_tsne[:len(real),0], data_tsne[:len(real),1], label="Real", alpha=0.5, c='blue')
    plt.scatter(data_tsne[len(real):,0], data_tsne[len(real):,1], label="Synthetic", alpha=0.5, c='red')
    plt.legend()
    plt.title(title)
    plt.show()
