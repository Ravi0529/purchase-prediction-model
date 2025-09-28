# Purchase Prediction from Ads — Logistic Regression

## Overview

This project builds a binary classifier to predict whether a user purchases a product after seeing a social network advertisement. It uses Logistic Regression on the classic `Social_Network_Ads` dataset and demonstrates:

- **EDA**: Distribution checks, correlations, and visualizations
- **Preprocessing**: Feature scaling and feature engineering
- **Modeling**: Baseline logistic regression on numeric features
- **Feature Engineering for Non‑Linearity**: Binning salary into quantiles and one‑hot encoding to capture non‑linear effects
- **Evaluation**: Accuracy, confusion matrix, and classification report on train/test splits

## Repository Structure

- `purchase_prediction_from_ads.ipynb`: End‑to‑end analysis and modeling notebook
- `data/Social_Network_Ads.csv`: Dataset used in the notebook
- `requirements.txt`: Python dependencies to reproduce the notebook
- `README.md`: This file

## Dataset

`data/Social_Network_Ads.csv` with 400 rows and 5 columns:

- `User ID` (int)
- `Gender` (Male/Female)
- `Age` (int)
- `EstimatedSalary` (int)
- `Purchased` (0/1 target)

There are no missing values. The target distribution is moderately imbalanced but not extreme.

## Approach

1. **Exploratory Data Analysis (EDA)**

   - Head/Info/Describe, duplicate checks
   - Scatter plots, box plots vs. `Purchased`
   - Crosstab for `Gender` vs. `Purchased`
   - Correlation heatmaps

2. **Baseline Modeling**

   - Features: `Age`, `EstimatedSalary`
   - Scaling: `StandardScaler` on features
   - Model: `LogisticRegression`
   - Split: `train_test_split(test_size=0.2, random_state=42)`

3. **Non‑Linearity Handling (Feature Engineering)**
   - Bin `EstimatedSalary` into 8 quantiles using `pd.qcut`
   - Compute purchase rate and log‑odds per bin to verify non‑linear relationship
   - Encode bins:
     - Ordinal encode for intermediate representation
     - One‑hot encode final feature set to avoid imposing linear order distances
   - Final features drop `User ID`, `Gender`, `EstimatedSalary` and retain one‑hot bin indicators (plus `Age` was not included in the engineered set per notebook’s final section)

## Results

- **Baseline (scaled `Age` + `EstimatedSalary`)**

  - Train accuracy: ~0.841
  - Test accuracy: ~0.863
  - Test report (abridged): good precision for both classes; class 1 recall ~0.68

- **With Salary Bins + One‑Hot**
  - Train accuracy: ~0.900
  - Test accuracy: ~0.950
  - Test confusion matrix: TN=50, FP=2, FN=2, TP=26
  - Balanced improvements across precision/recall for both classes

Interpretation: transforming income into binned indicators captures a non‑linear relationship between salary and purchase likelihood, substantially improving generalization.

## How to Run

1. **Clone or open the project**
2. **Create and activate a virtual environment** (recommended)
   - Windows (PowerShell or CMD):
     - `python -m venv .venv`
     - `.venv\\Scripts\\activate`
   - Git Bash:
     - `python -m venv .venv`
     - `source .venv/Scripts/activate`
3. **Install dependencies**
   - `pip install -r requirements.txt`
4. **Launch Jupyter**
   - `jupyter notebook` or `jupyter lab`
5. **Open and run** `purchase_prediction_from_ads.ipynb` cell‑by‑cell

### Requirements

Key packages (see full versions in `requirements.txt`):

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `matplotlib`, `seaborn`
- Jupyter stack: `ipykernel`, `jupyter_client`, `jupyter_core`

### Reproducibility

- Random state is fixed at `42` for the train/test split
- The dataset is included in `data/`
- Results above were obtained by running the notebook with the pinned `requirements.txt`

### Notes and Extensions

- You can experiment with adding `Age` back into the final engineered feature set, polynomial features, or interaction terms
- Try other models (e.g., tree‑based) for comparison
- Consider calibration curves or ROC‑AUC for additional evaluation

### Acknowledgments

The `Social_Network_Ads` dataset is a well‑known toy dataset commonly shared on learning platforms (e.g., Kaggle/Udemy variants). It is used here purely for educational purposes.
