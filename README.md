# 🧠 Legal Clause Similarity Detection

A Deep Learning project implementing and comparing **Siamese BiLSTM** and **BiLSTM with Additive Attention** architectures for identifying semantic similarity between legal clauses using **PyTorch**.

Developed by **Muhammad Moiz Sajjad** (2025)

---

## 📘 Overview

This project explores deep learning methods for detecting whether two legal clauses convey the same meaning — a key challenge in **legal document automation** and **contract analysis**.

Two models were implemented and compared:

1. **Model A:** Siamese BiLSTM  
2. **Model B:** BiLSTM with Additive Attention  

Both models were trained and evaluated on a Kaggle dataset of legal clauses labeled by type, using strong supervision.  
The results show that both architectures achieve near-perfect accuracy, with the attention-based model providing slightly higher interpretability.

---

## 🧩 Dataset and Preprocessing

- **Dataset:** 395 CSV files containing legal clauses grouped by clause type.  
- Each file corresponds to a specific legal concept (e.g., *Payment*, *Vacation*, *Security*).  
- Clauses from the same type are labeled as **1 (similar)**, and clauses from different types are labeled as **0 (not similar)**.

### **Preprocessing Steps**
1. Combined all 395 CSVs into a single dataset.  
2. Cleaned the text and removed duplicates or short entries.  
3. Assigned unique clause IDs using MD5 hashing to ensure **no text leakage** between splits.  
4. Split the data **70% train / 15% validation / 15% test**, stratified by category.  
5. Built a vocabulary (min frequency = 2, max vocab size = 40,000).  
6. Tokenized each clause and converted it to padded integer sequences.  
7. Created balanced positive and negative clause pairs for each dataset split.

---

## 🏗️ Model Architectures

### **Model A – Siamese BiLSTM**
- Twin BiLSTM networks share weights to encode two clauses into semantic embeddings.  
- Uses **max-pooling** to capture global contextual features.  
- The outputs of both branches are compared using:
  - Absolute difference |A−B|
  - Element-wise product A×B  
  - Concatenation [A, B]  
- These combined features are passed through fully connected layers to output a similarity score (Sigmoid).  

### **Model B – BiLSTM + Additive Attention**
- Extends Model A by adding **Bahdanau-style Additive Attention** after the BiLSTM layer.  
- Attention learns to assign importance weights to each token, enabling the model to focus on critical parts of the clause (e.g., legal action verbs, obligations).  
- Improves interpretability and robustness on longer clauses.

---

## ⚙️ Training Configuration

| Parameter | Value |
|:--|:--|
| Epochs | 6 |
| Batch Size | 64 |
| Sequence Length | 384 |
| Optimizer | Adam |
| Learning Rate | 2e-3 |
| Dropout | 0.2 |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Device | Auto-detect (GPU/CPU) |

Both models were trained for 6 epochs, converging around epoch 3–4 with consistently high validation accuracy.

---

## 📈 Training and Validation

Both networks showed smooth and stable training:
- Rapid convergence in early epochs (loss stabilized around 0.01).  
- Validation and training F1 scores remained above 0.99 throughout.  
- No signs of overfitting or divergence.  

The attention-based model had slightly longer epoch times due to the attention computation overhead but achieved slightly better F1 and interpretability.

---

## 📊 Evaluation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Avg Epoch Time |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Siamese BiLSTM (A)** | 0.997 | 0.997 | 0.998 | **0.998** | 1.000 | 1.000 | ~10.3s |
| **BiLSTM + Attention (B)** | 0.998 | 0.999 | 0.999 | **0.999** | 0.999 | 0.999 | ~11.6s |

**Interpretation:**
- Both models achieved near-perfect accuracy on the test set.  
- Model B performs marginally better in F1 and offers token-level interpretability.  
- Training efficiency remains excellent for both architectures.

---

## 💬 Qualitative Results

### ✅ Correct Predictions
- Clauses like *Vacation Policy* and *Employee Leave* were correctly recognized as semantically equivalent despite different wording.  
- Clauses from unrelated sections (e.g., *Security* vs *WITNESSETH*) were correctly identified as dissimilar.  
- The models successfully learned contextual cues beyond surface-level text similarity.

### ⚠️ Incorrect Predictions
- Clauses with similar structure but different semantics (e.g., *No Solicitation* vs *No Assignment*) caused false positives.  
- Overlap in legal phraseology (“Payment of ...”) occasionally confused the network.  
- Attention mechanism helped mitigate such cases by weighting discriminative tokens more strongly.

---

## 🧠 Discussion

Both architectures demonstrate exceptional performance due to:
- Clear labeling and well-separated clause categories.  
- Effective text encoding via BiLSTMs.  
- Balanced positive and negative training pairs.  

**Model Comparison:**
- The **Siamese BiLSTM** provides strong baseline accuracy with minimal complexity.  
- The **Attention-enhanced BiLSTM** improves interpretability by showing which words influenced similarity decisions the most.  

**Key Observations:**
- Over 99% accuracy indicates strong internal consistency of the dataset.  
- Small gains from attention show its utility in complex or lengthy legal clauses.  
- Real-world legal text (less standardized) would benefit more from attention or transformer-based methods.

---

## 🏁 Conclusion

This project successfully implements and evaluates two deep learning architectures for legal clause similarity detection.  
Both models achieve near-perfect performance, validating the power of sequence models for semantic textual similarity tasks.

**Key Takeaways:**
- Siamese BiLSTMs can model legal clause semantics effectively.  
- Additive Attention enhances model focus and interpretability.  
- These architectures form a strong baseline for legal document analysis and contract automation systems.

---

## ⚙️ Setup and Usage

### **Installation**
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
