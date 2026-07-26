# Spam SMS Detection

This repository documents the development of a text classification pipeline for detecting spam SMS messages — from data preparation and model selection through adversarial testing and root-cause debugging of the model's failure modes.

The project follows a complete cycle: build a pipeline that performs well on paper, stress-test it against inputs designed to break it, then diagnose exactly why it fails using the model's own learned parameters.

## Project Structure

```
SpamSMS/
│
├── SpamSMS.ipynb          # Full pipeline: EDA, preprocessing, model comparison, adversarial testing, debugging
├── spam_words_list.csv    # Exported SVM feature weights, used in the debugging phase
├── README.md
└── (v2 fixes in progress...)
```

---

## Project Overview

**Notebook:** `SpamSMS.ipynb`
**Purpose:** Classify SMS messages as spam or legitimate ("ham") using classical machine learning, with particular attention to *why* the final model succeeds or fails on specific inputs, not just its aggregate accuracy.

**Highlights:**

* Correct **train/test split before vectorization**, avoiding data leakage into the TF-IDF vocabulary.
* Systematic **comparison of four classifiers** — Naive Bayes, Logistic Regression, Linear SVM, and MLP — evaluated on the precision/recall tradeoff between the two classes, not on accuracy alone.
* **Adversarial testing** of the selected model against hand-written inputs designed to resemble real spam.
* **Root-cause debugging** of the resulting failures, using the model's own feature weights rather than guesswork.

---

## Pipeline Stages

| Stage | Purpose |
|---|---|
| **Data cleaning** | Checked for missing values; removed 403 duplicate records from 5,572 total. |
| **Preprocessing** | Lowercased text, stripped punctuation, removed stopwords. |
| **Train/test split** | Performed before vectorization to prevent data leakage. |
| **Vectorization** | TF-IDF, fit on the training set only. |
| **Model comparison** | Trained and evaluated four classifiers on the ham/spam tradeoff. |
| **Adversarial testing** | Ran the selected model against new, hand-written messages. |
| **Debugging** | Traced failures to their root causes via SVM feature weights. |

---

## Model Comparison

| Model | Behavior |
|---|---|
| Complement Naive Bayes | High spam recall, but too many false positives — flags real messages as spam. |
| Logistic Regression | Opposite failure mode — misses too many actual spam messages. |
| **Linear SVM (selected)** | Best balance of precision and recall across both classes. |
| MLP (Neural Network) | Marginally better spam recall than SVM, but not worth the added training cost and complexity for this dataset size. |

**Linear SVM was selected** as the final model based on this balance, not on accuracy in isolation.

---

## Debugging: Where the Model Breaks

After training, the pipeline was tested against new messages written to resemble real-world spam. Two failures emerged:

**Case 1 — `"Congrats! U w1n a phn!"`**
Predicted: not spam.
The word "win" carries one of the highest spam-indicative weights the model learned. However, "w1n" is a distinct token from "win" after vectorization — the model has no mechanism to relate leetspeak substitutions to their standard-spelling equivalents.

**Case 2 — `"SALE! Buy 1 Get 1 Free!"`**
Predicted: not spam.
Three compounding factors were identified:
1. "Free" carries a comparatively low spam weight on its own.
2. "Get" is removed by the stopword list.
3. The vectorizer discards single-digit tokens like "1."

After these reductions, the message effectively becomes "sale buy free" to the model — too weak a signal to trigger a spam classification.

**Investigation method:** the SVM's learned coefficients were extracted and mapped to their corresponding vocabulary terms (`spam_words_list.csv`), then sorted by weight to identify which words the model treats as spam-indicative. This confirmed both root causes directly, rather than relying on assumption.

---

## Planned Fix (v2)

* Reduce over-aggressive preprocessing — retain digits, reconsider the stopword list.
* Add basic text normalization for common leetspeak substitutions (1→i, 3→e, 0→o).
* Re-test against the same adversarial examples to confirm the fix resolves both failure cases.

---

## Setup & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/TasnimTamanna02/SpamSMS.git
   cd SpamSMS
   ```

2. **Dataset**
   [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), UCI Machine Learning Repository. 

3. **Run in Jupyter or Google Colab**
   The notebook currently loads data via a Google Drive path (Colab-specific). To run locally, update the file path in the data import cell to point to your local copy of the dataset.

---

## Tech Stack

Python, pandas, scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Linear SVM, MLP), NLTK, matplotlib, seaborn

---

## Author

**Tasnim Tamanna**
*Computer Science Graduate | Aspiring AI Researcher | Exploring Machine Learning and Agentic AI Systems*
