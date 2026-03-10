# 🔧 Industrial Machine Failure Classifier

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-91.2%25-success?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-0.975-blue?style=flat-square)

> A binary classification model that predicts industrial equipment failure from sensor data — before the machine breaks down.

---

## 📊 Live Dashboard

👉 **[View Interactive Dashboard](https://abhyuday580.github.io/machine-failure-classifier)**

---

## 🎯 Project Overview

Industrial machines fail unexpectedly, causing costly downtime and safety risks. This project uses real sensor data from the **AI4I 2020 Predictive Maintenance Dataset** to build a model that flags machines likely to fail — giving engineers time to act before a breakdown occurs.

The core challenge: only **3.4% of records are failures** (highly imbalanced). A naive model ignoring this would be useless. This project solves that with **SMOTE** and optimises for **Recall** — because missing a real failure is far worse than a false alarm.

---

## 📈 Results

| Metric | Value |
|---|---|
| ROC-AUC Score | **0.975** |
| Failure Recall | **91.2%** |
| Failure Precision | 34.1% |
| Accuracy | 94.0% |
| Failures Caught (test set) | **62 out of 68** |

---

## 🗂️ Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository

- 10,000 records, 14 columns
- 339 real machine failures (3.4%)
- 5 sensor features used for prediction

| Feature | Description |
|---|---|
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Machine operating temperature |
| Rotational speed [rpm] | Spindle rotation speed |
| Torque [Nm] | Rotational force applied |
| Tool wear [min] | Cumulative tool usage time |

---

## 🧠 Key Findings

- **Torque** is the #1 predictor of failure (35.2% importance)
- **Tool Wear** is the #2 predictor (21.7% importance)
- Machines with torque above **67.5 Nm** fail at a **91.7% rate**
- Machines with tool wear above **221 min** fail at a **26.8% rate**
- **Low-quality (L-type)** machines fail 87% more often than high-quality (H-type)

---

## ⚙️ ML Pipeline
```
Raw Data (10,000 rows)
        |
Feature Selection (5 sensor columns)
        |
Train / Test Split (80% / 20%, stratified)
        |
StandardScaler (normalise sensor ranges)
        |
SMOTE (failures: 271 -> 2,576 synthetic samples)
        |
Random Forest Classifier (150 trees, depth 10)
        |
Threshold Tuning (0.5 -> 0.30 for high recall)
        |
Evaluation + Visualisation
```

---

## 🔬 SMOTE — Handling Class Imbalance

SMOTE was implemented **from scratch** without using the imbalanced-learn library.

| | Before SMOTE | After SMOTE |
|---|---|---|
| No Failure | 7,729 | 7,729 |
| Failure | 271 | 2,576 |

---

## 📁 Project Structure
```
machine-failure-classifier/
├── machine_failure_classifier.py   # Main ML pipeline
├── ai4i2020.csv                    # Dataset
├── index.html                      # Interactive dashboard
├── class_distribution.png         # SMOTE before/after chart
├── feature_importance.png         # Top failure drivers chart
├── confusion_matrix.png           # Prediction breakdown
├── roc_curve.png                  # Model performance curve
└── README.md                      # This file
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/abhyuday580/machine-failure-classifier.git
cd machine-failure-classifier
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn matplotlib seaborn
```

**3. Run the classifier**
```bash
python machine_failure_classifier.py
```

**4. View the dashboard**

Open `index.html` in any browser — no server needed.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| Pandas | Data loading and analysis |
| NumPy | SMOTE implementation |
| Scikit-Learn | ML model, scaling, evaluation |
| Matplotlib + Seaborn | Chart generation |
| HTML + Chart.js | Interactive dashboard |

---

## 👤 Author

**Abhyuday**
B.Tech Metallurgical & Materials Engineering — NIT Raipur

[![GitHub](https://img.shields.io/badge/GitHub-abhyuday580-black?style=flat-square&logo=github)](https://github.com/abhyuday580)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
