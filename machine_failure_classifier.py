"""
=======================================================
  Industrial Machine Failure Classifier
  Dataset : AI4I 2020 Predictive Maintenance (Real)
  Goal    : Predict whether a machine will fail (0 / 1)
  Tools   : Python, Pandas, Scikit-Learn, Matplotlib
=======================================================

Steps in this project:
  1. Load & explore the dataset
  2. Clean and prepare data
  3. Handle class imbalance using SMOTE
  4. Train a Random Forest model
  5. Evaluate with focus on Recall
  6. Plot: class balance, feature importance,
           confusion matrix, ROC curve
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
)

# Set a seed so results are the same every run
SEED = 42
np.random.seed(SEED)


# =============================================================================
# STEP 1 - Load the dataset
# =============================================================================

print("=" * 55)
print("  STEP 1 - Loading Dataset")
print("=" * 55)

df = pd.read_csv("ai4i2020.csv")

# The CSV has a hidden BOM character in the first column name - clean it up
df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

print(f"\nDataset shape  : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:\n  {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())


# =============================================================================
# STEP 2 - Explore & Clean the data
# =============================================================================

print("\n" + "=" * 55)
print("  STEP 2 - Data Exploration & Cleaning")
print("=" * 55)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nClass distribution (Machine failure):")
counts = df["Machine failure"].value_counts()
print(f"  No Failure (0) : {counts[0]}  ({counts[0]/len(df)*100:.1f}%)")
print(f"  Failure    (1) : {counts[1]}  ({counts[1]/len(df)*100:.1f}%)")
print("\n  >> This is imbalanced! Only 3.4% are failures.")
print("  >> We will fix this with SMOTE oversampling.")

print("\nBasic statistics for sensor columns:")
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
print(df[sensor_cols].describe().round(2).to_string())

# We only use sensor columns as features (drop IDs, product type, sub-failures)
FEATURES = sensor_cols
TARGET   = "Machine failure"

X = df[FEATURES].values   # shape: (10000, 5)
y = df[TARGET].values     # shape: (10000,)


# =============================================================================
# STEP 3 - Train / Test Split + Feature Scaling
# =============================================================================

print("\n" + "=" * 55)
print("  STEP 3 - Train/Test Split")
print("=" * 55)

# stratify=y keeps the same failure % in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Train failures   : {y_train.sum()}")
print(f"Test failures    : {y_test.sum()}")

# Scale features - puts them all on the same scale (mean=0, std=1)
# Important: fit ONLY on training data, then apply to test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# =============================================================================
# STEP 4 - SMOTE (Synthetic Minority Over-sampling Technique)
#
# Problem : Only ~3% of rows are failures. A model that predicts "no failure"
#           every single time would be 97% accurate - but completely useless!
#           We actually need to CATCH failures, so we need the model to see
#           more failure examples during training.
#
# Solution: SMOTE creates new artificial failure samples by blending existing
#           ones together. It does NOT just copy - it creates new in-between
#           points so the model learns a richer picture of what failure looks like.
# =============================================================================

print("\n" + "=" * 55)
print("  STEP 4 - SMOTE (Fixing Class Imbalance)")
print("=" * 55)


def apply_smote(X, y, target_minority_ratio=0.25, k_neighbors=5, random_state=42):
    """
    A clean, from-scratch implementation of SMOTE.

    Parameters:
        X                     : feature array (already scaled)
        y                     : label array (0 or 1)
        target_minority_ratio : what fraction of the data should be failures
        k_neighbors           : how many nearest neighbours to consider
        random_state          : for reproducibility

    Returns:
        X_new, y_new : augmented arrays with synthetic failure samples added
    """
    rng = np.random.default_rng(random_state)

    X_minority = X[y == 1]   # all failure rows
    n_min      = len(X_minority)
    n_maj      = (y == 0).sum()

    # Formula: solve for how many extra samples we need
    # n_min_new / (n_maj + n_min_new) = target_minority_ratio
    n_needed = int((target_minority_ratio * n_maj) / (1 - target_minority_ratio)) - n_min

    if n_needed <= 0:
        print("  Class is already balanced enough - skipping SMOTE.")
        return X, y

    print(f"\n  Generating {n_needed} synthetic failure samples ...")
    synthetic_X = []

    for _ in range(n_needed):
        # 1. Pick a random minority sample (the "anchor")
        i      = rng.integers(0, n_min)
        anchor = X_minority[i]

        # 2. Find its k nearest neighbours among other minority samples
        dists = np.linalg.norm(X_minority - anchor, axis=1)
        dists[i] = np.inf                        # ignore self
        k_nearest = np.argsort(dists)[:k_neighbors]

        # 3. Pick one of those neighbours randomly
        neighbour = X_minority[rng.choice(k_nearest)]

        # 4. Create a new point on the line segment between anchor and neighbour
        #    alpha=0 -> exact copy of anchor, alpha=1 -> exact copy of neighbour
        alpha     = rng.random()
        new_point = anchor + alpha * (neighbour - anchor)
        synthetic_X.append(new_point)

    X_synthetic = np.array(synthetic_X)
    y_synthetic = np.ones(n_needed, dtype=int)

    # Combine original data + synthetic failure samples
    X_new = np.vstack([X, X_synthetic])
    y_new = np.concatenate([y, y_synthetic])

    return X_new, y_new


# Store original counts for the plot
n_before_0 = (y_train == 0).sum()
n_before_1 = (y_train == 1).sum()

print(f"  Before SMOTE - No Failure: {n_before_0:,}  |  Failure: {n_before_1}")

X_train_res, y_train_res = apply_smote(
    X_train_scaled, y_train,
    target_minority_ratio=0.25,
    k_neighbors=5,
    random_state=SEED,
)

n_after_0 = (y_train_res == 0).sum()
n_after_1 = (y_train_res == 1).sum()
print(f"  After  SMOTE - No Failure: {n_after_0:,}  |  Failure: {n_after_1:,}")


# =============================================================================
# STEP 5 - Train the Model
# =============================================================================

print("\n" + "=" * 55)
print("  STEP 5 - Training Random Forest Classifier")
print("=" * 55)
print("""
  Why Random Forest?
    - Builds many decision trees (like getting multiple opinions)
    - Combines their votes for a final prediction
    - Works well with tabular/sensor data
    - Gives us feature importances for free
""")

model = RandomForestClassifier(
    n_estimators=150,        # number of decision trees
    max_depth=10,            # max depth of each tree (limits overfitting)
    class_weight="balanced", # penalises missing failures more than false alarms
    random_state=SEED,
    n_jobs=-1,               # use all CPU cores to train faster
)

model.fit(X_train_res, y_train_res)
print("  Model trained successfully!")


# =============================================================================
# STEP 6 - Evaluate the Model
#
# Most important metric here: RECALL for the "Failure" class
#
#   Recall = True Positives / (True Positives + False Negatives)
#          = "Of all the real failures, how many did we catch?"
#
# We prefer a false alarm (flagging a healthy machine) over MISSING a real
# failure - a missed failure could mean equipment breakdown or injury.
# =============================================================================

print("\n" + "=" * 55)
print("  STEP 6 - Model Evaluation")
print("=" * 55)

# Get probability scores (a number between 0 and 1 for each sample)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Lower the threshold from default 0.5 -> 0.3 to flag more failures
# (this increases recall at the cost of some precision - acceptable here)
THRESHOLD = 0.30
y_pred    = (y_proba >= THRESHOLD).astype(int)

print(f"\n  Threshold used : {THRESHOLD}")
print(f"  (Default is 0.5; lowering it makes the model more sensitive to failures)\n")
print("  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
auc       = roc_auc_score(y_test, y_proba)

print(f"  ROC-AUC Score     : {auc:.4f}  (1.0 = perfect, 0.5 = random guess)")
print(f"  Failure Recall    : {recall:.3f}  ->  {recall*100:.1f}% of real failures caught")
print(f"  Failure Precision : {precision:.3f}")


# =============================================================================
# STEP 7 - Visualisations
# =============================================================================

print("\n" + "=" * 55)
print("  STEP 7 - Saving Plots")
print("=" * 55)


# ── Plot 1: Class Distribution before vs after SMOTE ─────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Class Balance: Before vs After SMOTE",
    fontsize=14, fontweight="bold", y=1.02,
)

plot_data = [
    (n_before_0, n_before_1, "Before SMOTE (Training Set)"),
    (n_after_0,  n_after_1,  "After SMOTE (Training Set)"),
]

for ax, (n0, n1, title) in zip(axes, plot_data):
    bars = ax.bar(
        ["No Failure", "Failure"],
        [n0, n1],
        color=["#4A90D9", "#E74C3C"],
        edgecolor="black",
        width=0.45,
    )
    for bar, val in zip(bars, [n0, n1]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(n0, n1) * 0.015,
            f"{val:,}",
            ha="center", fontsize=12, fontweight="bold",
        )
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(n0, n1) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> class_distribution.png")


# ── Plot 2: Feature Importance ────────────────────────────────────────────────

importances = model.feature_importances_
order       = np.argsort(importances)[::-1]   # highest first

short_names = {
    "Air temperature [K]"      : "Air Temp",
    "Process temperature [K]"  : "Process Temp",
    "Rotational speed [rpm]"   : "Rot. Speed",
    "Torque [Nm]"              : "Torque",
    "Tool wear [min]"          : "Tool Wear",
}
labels = [short_names[FEATURES[i]] for i in order]
colors = ["#E74C3C" if i < 2 else "#4A90D9" for i in range(len(order))]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, importances[order], color=colors, edgecolor="black", width=0.5)

for bar, val in zip(bars, importances[order]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.3f}",
        ha="center", fontsize=10, fontweight="bold",
    )

plt.ylabel("Importance Score", fontsize=11)
plt.title(
    "Feature Importance - What Drives Machine Failures?",
    fontsize=13, fontweight="bold",
)
plt.annotate(
    "Red = top 2 drivers",
    xy=(0.97, 0.95), xycoords="axes fraction",
    ha="right", fontsize=9.5, color="#E74C3C",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", alpha=0.7),
)
plt.ylim(0, max(importances) * 1.22)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> feature_importance.png")


# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────

cm     = confusion_matrix(y_test, y_pred)
labels_cm = ["No Failure", "Failure"]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_cm,
    yticklabels=labels_cm,
    linewidths=1,
    linecolor="gray",
    annot_kws={"size": 16, "weight": "bold"},
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
fig.text(
    0.5, -0.06,
    "TN (top-left) = correct no-failure   |   TP (bottom-right) = failure correctly caught\n"
    "FP (top-right) = false alarm   |   FN (bottom-left) = MISSED failure  ← minimise this",
    ha="center", fontsize=8.5, color="#555",
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> confusion_matrix.png")


# ── Plot 4: ROC Curve ─────────────────────────────────────────────────────────

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#E74C3C", lw=2.5,
         label=f"Our model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1.5,
         label="Random guess (AUC = 0.500)")
plt.fill_between(fpr, tpr, alpha=0.07, color="#E74C3C")

# Mark the dot for our chosen threshold on the curve
chosen_idx = np.argmin(np.abs(thresholds - THRESHOLD))
plt.scatter(
    fpr[chosen_idx], tpr[chosen_idx],
    color="black", zorder=5, s=90,
    label=f"Our threshold={THRESHOLD} (Recall={tpr[chosen_idx]:.2f})",
)

plt.xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Recall)", fontsize=11)
plt.title("ROC Curve", fontsize=13, fontweight="bold")
plt.legend(fontsize=9.5, loc="lower right")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> roc_curve.png")


# =============================================================================
# Final Summary
# =============================================================================

top2 = [short_names[FEATURES[order[0]]], short_names[FEATURES[order[1]]]]

print("\n" + "=" * 55)
print("  PROJECT SUMMARY")
print("=" * 55)
print(f"""
  Dataset        : AI4I 2020 - 10,000 samples, 339 real failures
  Model          : Random Forest (150 trees, max depth 10)
  Imbalance fix  : SMOTE - failures {n_before_1} -> {n_after_1:,} training samples

  Evaluation on {len(y_test)} test samples:
    ROC-AUC        : {auc:.3f}
    Failure Recall : {recall:.3f}  ({recall*100:.1f}% of failures caught)
    Precision      : {precision:.3f}

  Top failure predictors:
    1st -> {top2[0]}
    2nd -> {top2[1]}

  Output charts:
    class_distribution.png  - before vs after SMOTE
    feature_importance.png  - which sensors matter most
    confusion_matrix.png    - prediction breakdown
    roc_curve.png           - model performance curve
""")
