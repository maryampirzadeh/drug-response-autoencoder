# Drug Response Prediction using Multi-Omics Data with Autoencoders

## 📌 Overview

This project focuses on **predicting cancer drug response** using **multi-omics data** and **deep learning models**.
By integrating heterogeneous biological data (gene expression, mutations, copy number, drug descriptors, etc.), we build a robust pipeline based on:

* Autoencoders for feature representation learning
* Latent space fusion of cell and drug features
* A neural network classifier for response prediction

The model achieves **>98% AUPRC**, demonstrating strong predictive performance for drug sensitivity and resistance.

---

## 🧬 Datasets

This project uses publicly available pharmacogenomics datasets:

* **CTRP (Cancer Therapeutics Response Portal)**
* **GDSC (Genomics of Drug Sensitivity in Cancer)**
* **CCLE (Cancer Cell Line Encyclopedia)**

### Data Modalities

* Cell features:

  * Gene expression
  * Copy number variation
  * DNA methylation
  * Mutation data
* Drug features:

  * Drug descriptors
  * Chemical fingerprints
  * Drug targets
  * Compound properties

---

## 🧠 Model Architecture

### 1. Autoencoders

* Separate autoencoders for:

  * Cell features
  * Drug features
* Learn compressed latent representations

### 2. Latent Space Fusion

* Concatenate:

  ```
  cell_latent + drug_latent → combined_latent
  ```

### 3. Classifier

* Multi-layer perceptron (MLP)
* Binary classification:

  * 0 → Resistant
  * 1 → Sensitive

---

## ⚙️ Pipeline

1. Load and preprocess multi-omics datasets
2. Match cell lines and drugs across all modalities
3. Normalize features
4. Train:

   * Cell autoencoder
   * Drug autoencoder
5. Generate latent representations
6. Train classifier on combined latent space
7. Evaluate performance (AUC, AUPRC, Accuracy, etc.)

---

## 📊 Results

* **AUPRC:** > 98%
* **AUC:** High performance across validation and test sets
* Strong ability to distinguish between drug-sensitive and resistant samples

---

## 🗂️ Project Structure

```
├── data_processing.py        # Data loading & preprocessing
├── train_cell_autoencoder.py
├── train_drug_autoencoder.py
├── train_classifier.py
├── train_deepdra.py
├── evaluation.py
├── main.py                   # Full pipeline
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### 2. Update dataset paths

Modify paths inside scripts to point to your local dataset folders.

### 3. Run full pipeline

```bash
python main.py
```

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* AUPRC

---

## 🔬 Key Contributions

* Integration of **multi-omics data** for drug response prediction
* Use of **autoencoders for dimensionality reduction**
* Joint modeling of **cell and drug features**
* High-performance predictive modeling for cancer treatment response


This project is for research and educational purposes.
