
# 🫁 Lung Cancer Prediction & Risk Stratification

## 📌 Project Overview

This project develops a robust machine learning pipeline to predict the **risk of lung cancer** in patients based on a comprehensive set of demographic, lifestyle, environmental, and clinical factors.

By analyzing 5,000 patient records, we built a system optimized for **Recall (Sensitivity)** to minimize false negatives—a critical requirement in medical diagnostics. The final model utilizes an **SGD Classifier**, achieving a Recall of **~98.4%**, effectively identifying high-risk individuals for early screening.

### 🔍 Key Highlights

* **End-to-End Pipeline:** From raw data EDA to deploying a Voting Ensemble.
* **Feature Engineering:** Created interaction terms (e.g., `Pollution * Smoker`) that uncovered hidden risk multipliers.
* **Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to address the 75:25 class imbalance.
* **Extensive Benchmarking:** Trained and compared **17 different algorithms**, including Logistic Regression, Random Forests, XGBoost, SVMs, and Neural Networks (MLP).

---

## 📂 Repository Structure

Based on the project architecture, the files are organized as follows:

```bash
lung_cancer_prediction/
├── data/
│   ├── lung_cancer.csv                  # Original Dataset
│   └── pre_processed_data/              # Generated during Phase 2
│       ├── train_processed.csv          # Scaled & Balanced Training Set
│       └── test_processed.csv           # Scaled Testing Set (Unseen data)
├── 1_exploratory_data_analysis.ipynb    # EDA: Visualizing correlations & distributions
├── 2_data_preprocessing.ipynb           # Cleaning, Scaling, Feature Engineering, SMOTE
├── 3_model.ipynb                        # Model Training, Hyperparameter Tuning, Evaluation
├── catboost_info/                       # CatBoost training logs
├── LICENSE                              # Apache 2.0 License
└── README.md                            # Project Documentation

```

---

## 📊 Dataset Details

The dataset contains patient-level information specifically curated for cancer risk analysis.

* **Source:** [Kaggle - Lung Cancer Prediction Dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/dhrubangtalukdar/lung-cancer-prediction-dataset/data)
* **Size:** 5,000 Rows, 30 Columns.
* **Target:** `lung_cancer_risk` (0 = Low, 1 = High).

**Key Features:**

* **Lifestyle:** `smoker`, `pack_years` (Cumulative exposure), `diet_quality`.
* **Clinical:** `oxygen_saturation`, `fev1_x10` (Lung capacity), `genetic_markers`.
* **Environment:** `air_pollution_index`, `occupational_exposure`.

---

## 🚀 Methodology

### Phase 1: Exploratory Data Analysis (EDA)

We conducted a deep dive into the data to understand risk factors.

* **Smoking Dominance:** Visualizations confirmed that 100% of high-risk patients in this dataset had a history of smoking.
* **Pollution Interaction:** We discovered that air pollution barely affects non-smokers but significantly amplifies risk for smokers.
* **Symptom Clustering:** High-risk patients showed a distinct prevalence of `shortness_of_breath` and `chronic_cough`, while `chest_pain` was surprisingly non-specific.

### Phase 2: Data Preprocessing

* **Logical Cleaning:** Enforced data integrity by ensuring non-smokers had `0` cigarettes/day.
* **Feature Engineering:**
* Created `pollution_smoker_interaction` to capture the multiplier effect.
* Aggregated respiratory symptoms into a `symptom_score`.


* **Scaling & Balancing:** Applied **StandardScaler** to normalize numerical ranges and **SMOTE** to balance the training data, ensuring the model didn't just memorize the majority class.

### Phase 3: Model Modeling & Evaluation

We prioritized **Recall** (Sensitivity) and the **F2-Score** (which weights Recall higher than Precision) to ensure we don't miss positive cancer cases.

**Models Tested:**

* **Linear:** Logistic Regression, SGD Classifier, LDA/QDA.
* **Trees/Ensembles:** Random Forest, AdaBoost, XGBoost, LightGBM, CatBoost.
* **Deep Learning:** Multi-Layer Perceptron (MLP).
* **Voting:** A Soft Voting Classifier combining the top 3 performers.

---

## 🏆 Results & Findings

Contrary to the trend where boosting trees (XGBoost) usually win, **Linear Models** performed exceptionally well, likely due to the strong linear correlation of `pack_years` with risk.

| Rank | Model | Recall (Sensitivity) | F2-Score |
| --- | --- | --- | --- |
| 🥇 | **SGD Classifier** | **98.39%** | **0.9715** |
| 🥈 | **Voting Classifier** | 97.99% | 0.9713 |
| 🥉 | **MLP (Neural Network)** | 96.79% | 0.9679 |
| 4 | **Logistic Regression** | 96.79% | 0.9563 |

* **Winning Model:** The `SGDClassifier` with ElasticNet penalty proved to be the most robust, effectively filtering noise while capturing the dominant risk signals.
* **Strategic Insight:** The "Davids" (simple linear models) beat the "Goliaths" (Complex Gradient Boosting), proving that for datasets with strong linear predictors, simpler models offer better interpretability and efficiency.

---

## 🛠️ Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/HrushiSanap/lung_cancer_prediction.git
cd lung_cancer_prediction

```


2. **Install dependencies:**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost lightgbm catboost imbalanced-learn

```


3. **Run the analysis:**
* Open `1_exploratory_data_analysis.ipynb` to view the data story.
* Run `2_data_preprocessing.ipynb` to generate the clean datasets.
* Run `3_model.ipynb` to train the models and see the final leaderboard.



---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🙏 Acknowledgments

* Dataset provided by **Dhrubang Talukdar** on Kaggle.
* Analysis supported by **Scikit-Learn** and **Imbalanced-Learn** documentation.