# CRBM-Based Data Augmentation Framework  
*Enhancing Data Efficiency and Diversity with Conditional Restricted Boltzmann Machines*

---

## 🔎 Overview

In many machine learning applications, data scarcity or class imbalance can limit the performance of models.  
This project proposes a **Conditional Restricted Boltzmann Machine (CRBM)** based framework for synthetic data generation and augmentation, which can be applied to both image and tabular datasets.  

> The key innovation is introducing a configurable hyperparameter that allows the model to toggle between standard RBM and CRBM modes, making the pipeline flexible and adaptable for various datasets.

---

## 🧩 Key Concepts  

### 1. **RBM (Restricted Boltzmann Machine)**  
An unsupervised probabilistic generative model that learns the joint distribution of visible and hidden units:
- Learns data distribution through Contrastive Divergence.
- Suitable for non-conditional generation.

### 2. **CRBM (Conditional RBM)**  
An extension of RBM that incorporates class-conditioning:
- Visible layer generation is conditioned on one-hot class vectors.
- Allows generating samples from specific classes.

### 3. **Adaptive Contrastive Divergence (A-CD)**  
Instead of fixed CD-k steps, the training process dynamically adjusts the number of Gibbs sampling steps based on training loss convergence.
- Faster convergence.
- Stabilizes learning in deep RBMs or CRBMs.

---

## 📊 Proposed Pipeline  

1. **Dataset Loading**  
   - Supports both image datasets (e.g., MNIST) and tabular datasets (via CSV).  
   
2. **Model Selection**  
   - Controlled by a hyperparameter `rbm_type` in the config file:
     - `RBM` or `CRBM`.
   
3. **Training**  
   - Trains RBM/CRBM with optional adaptive CD.
   - Utilizes PyTorch for tensor operations and GPU acceleration.

4. **Synthetic Data Generation**  
   - If CRBM is used, samples are generated class-conditionally.
   - Synthetic samples are visualized using t-SNE for inspection.

5. **Augmentation & Evaluation**  
   - Synthetic data is combined with real data to train downstream classifiers.
   - Performance improvement is evaluated.
   - KS-Tests are conducted to statistically assess distribution differences.