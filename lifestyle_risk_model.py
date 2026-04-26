
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
from sklearn.inspection import permutation_importance
from scipy import stats

import os
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SECTION 1 — DATA LOADING & EXPLORATION

print("=" * 70)
print("  LIFESTYLE DISEASE RISK PREDICTION — ML PIPELINE")
print("=" * 70)

print("\n[1/7] Loading and exploring dataset...")

df_raw = pd.read_csv('new_dataset_NHANES__1_.csv')
print(f"  Raw shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"  Columns: {list(df_raw.columns)}")

print("\n  --- Missing Value Summary ---")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0].to_string())

print("\n  --- Basic Statistics ---")
print(df_raw.describe().round(2).to_string())

# SECTION 2 — DATA PREPROCESSING & FEATURE ENGINEERING

print("\n[2/7] Preprocessing and feature engineering...")

df = df_raw.copy()

# --- Average the two BP readings ---
df['SBP'] = df[['BP UP', 'BP UP.1']].mean(axis=1)
df['DBP'] = df[['BP DOWN', 'BP DOWN.1']].mean(axis=1)

# --- Handle missing values ---
# Numeric columns: fill with median (robust to outliers)
num_cols = ['SBP', 'DBP', 'BMI', 'Weight', 'Height', 'Leg Length',
            'Arm Length', 'Arm Circumference', 'Waist Circumference', 'INDFMPIR']
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  Filled '{col}' NaN → median ({median_val:.2f})")

cat_cols = ['Alcohol Use', 'Average Drinks', 'DMDEDUC2', 'DMDMARTL', 'Health Insurance']
for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

df.drop(columns=['Alcohol Frequency', 'Sr', 'WTINT2YR', 'SDMVPSU', 'SDMVSTRA',
                  'BP UP', 'BP DOWN', 'BP UP.1', 'BP DOWN.1', 'DMDCITZN'], inplace=True)

# --- Feature Engineering ---
# BMI categories (WHO standard)
df['BMI_Category'] = pd.cut(df['BMI'],
    bins=[0, 18.5, 25, 30, 35, 100],
    labels=[0, 1, 2, 3, 4]  # underweight, normal, overweight, obese1, obese2+
).astype(int)

# BP Stage (ACC/AHA 2017 guidelines)
def bp_stage(row):
    sbp, dbp = row['SBP'], row['DBP']
    if sbp < 120 and dbp < 80:   return 0  # Normal
    elif sbp < 130 and dbp < 80: return 1  # Elevated
    elif sbp < 140 or dbp < 90:  return 2  # Stage 1 HTN
    else:                         return 3  # Stage 2 HTN

df['BP_Stage'] = df.apply(bp_stage, axis=1)

# Age groups
df['Age_Group'] = pd.cut(df['AGE'],
    bins=[0, 30, 45, 60, 200], labels=[0, 1, 2, 3]).astype(int)

# Waist-to-Height ratio (visceral obesity proxy)
df['WHtR'] = df['Waist Circumference'] / df['Height']

# Pulse Pressure (SBP - DBP)
df['Pulse_Pressure'] = df['SBP'] - df['DBP']

print(f"\n  Engineered features: BMI_Category, BP_Stage, Age_Group, WHtR, Pulse_Pressure")
print(f"  Dataset after preprocessing: {df.shape[0]} rows × {df.shape[1]} columns")

# SECTION 3 — TARGET VARIABLE CREATION

print("\n[3/7] Creating target variable (Disease Risk)...")

# Multi-factor risk score (based on clinical literature):
# Hypertension (Stage1+), Obesity (BMI≥30), Smoking, Age≥45, High WHtR
def compute_risk(row):
    score = 0
    if row['BP_Stage'] >= 2:            score += 1  # Hypertension
    if row['BMI'] >= 30:                score += 1  # Obesity
    if row['Smoking Status'] == 2:      score += 1  # Smoker (coded 2 in NHANES)
    if row['AGE'] >= 45:                score += 1  # Older age
    if row['WHtR'] > 0.5:              score += 1  # Abdominal obesity
    return score

df['Risk_Score'] = df.apply(compute_risk, axis=1)

# Binary target: High Risk = score ≥ 2
df['High_Risk'] = (df['Risk_Score'] >= 2).astype(int)

risk_counts = df['High_Risk'].value_counts()
print(f"  Low Risk  (0): {risk_counts.get(0, 0)} ({risk_counts.get(0,0)/len(df)*100:.1f}%)")
print(f"  High Risk (1): {risk_counts.get(1, 0)} ({risk_counts.get(1,0)/len(df)*100:.1f}%)")

# SECTION 4 — EXPLORATORY DATA ANALYSIS (EDA) WITH VISUALIZATIONS

print("\n[4/7] Running EDA and generating visualizations...")

plt.style.use('dark_background')
PALETTE = ['#06b6d4', '#ef4444', '#34d399', '#fbbf24', '#a78bfa', '#f472b6']

# --- FIGURE 1: EDA Overview ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#0f172a')
fig.suptitle('NHANES Dataset — Exploratory Data Analysis', fontsize=16, fontweight='bold', color='white', y=1.01)

# 1a. BMI Distribution
ax = axes[0, 0]
ax.set_facecolor('#1e293b')
ax.hist(df['BMI'], bins=40, color='#06b6d4', edgecolor='#0f172a', alpha=0.85)
ax.axvline(25, color='#fbbf24', linestyle='--', lw=1.5, label='Overweight (25)')
ax.axvline(30, color='#ef4444', linestyle='--', lw=1.5, label='Obese (30)')
ax.set_title('BMI Distribution', color='white', fontweight='bold')
ax.set_xlabel('BMI (kg/m²)', color='#9ca3af')
ax.legend(fontsize=8)
ax.tick_params(colors='#9ca3af')

# 1b. SBP vs DBP scatter
ax = axes[0, 1]
ax.set_facecolor('#1e293b')
colors = df['BP_Stage'].map({0:'#34d399', 1:'#fbbf24', 2:'#f97316', 3:'#ef4444'})
ax.scatter(df['SBP'], df['DBP'], c=colors, alpha=0.3, s=8)
ax.set_title('SBP vs DBP (BP Stage colored)', color='white', fontweight='bold')
ax.set_xlabel('Systolic BP', color='#9ca3af'); ax.set_ylabel('Diastolic BP', color='#9ca3af')
ax.tick_params(colors='#9ca3af')
for stage, color, label in [(0,'#34d399','Normal'),(1,'#fbbf24','Elevated'),(2,'#f97316','Stage 1'),(3,'#ef4444','Stage 2')]:
    ax.scatter([], [], c=color, label=label, s=30)
ax.legend(fontsize=8)

# 1c. Age Distribution by Risk
ax = axes[0, 2]
ax.set_facecolor('#1e293b')
for risk_val, color, label in [(0, '#34d399', 'Low Risk'), (1, '#ef4444', 'High Risk')]:
    subset = df[df['High_Risk'] == risk_val]['AGE']
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='none')
ax.set_title('Age Distribution by Risk Level', color='white', fontweight='bold')
ax.set_xlabel('Age (years)', color='#9ca3af')
ax.legend(fontsize=9)
ax.tick_params(colors='#9ca3af')

# 1d. Correlation Heatmap
ax = axes[1, 0]
ax.set_facecolor('#1e293b')
corr_cols = ['AGE', 'BMI', 'SBP', 'DBP', 'Waist Circumference', 'Arm Circumference', 'WHtR', 'High_Risk']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix), k=1)
sns.heatmap(corr_matrix, ax=ax, cmap='RdYlGn', center=0, annot=True, fmt='.2f',
            annot_kws={'size': 7}, linewidths=0.5, linecolor='#0f172a',
            cbar_kws={'shrink': 0.7})
ax.set_title('Correlation Matrix', color='white', fontweight='bold')
ax.tick_params(colors='#9ca3af', labelsize=8)

# 1e. Risk Score Distribution
ax = axes[1, 1]
ax.set_facecolor('#1e293b')
score_counts = df['Risk_Score'].value_counts().sort_index()
bars = ax.bar(score_counts.index, score_counts.values,
              color=['#34d399','#6ee7b7','#fbbf24','#f97316','#ef4444','#dc2626'][:len(score_counts)],
              edgecolor='#0f172a', linewidth=0.5)
ax.set_title('Risk Score Distribution (0–5)', color='white', fontweight='bold')
ax.set_xlabel('Risk Score', color='#9ca3af'); ax.set_ylabel('Count', color='#9ca3af')
ax.tick_params(colors='#9ca3af')
for bar, val in zip(bars, score_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val),
            ha='center', fontsize=9, color='white')

# 1f. BMI vs Waist Circumference
ax = axes[1, 2]
ax.set_facecolor('#1e293b')
ax.scatter(df['BMI'], df['Waist Circumference'],
           c=df['High_Risk'].map({0:'#34d399', 1:'#ef4444'}),
           alpha=0.3, s=8)
ax.set_title(f'BMI vs Waist Circumference (r={df["BMI"].corr(df["Waist Circumference"]):.2f})', color='white', fontweight='bold')
ax.set_xlabel('BMI', color='#9ca3af'); ax.set_ylabel('Waist (cm)', color='#9ca3af')
ax.tick_params(colors='#9ca3af')
ax.scatter([], [], c='#34d399', label='Low Risk', s=30)
ax.scatter([], [], c='#ef4444', label='High Risk', s=30)
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig1_eda_overview.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  Saved: fig1_eda_overview.png")

# =============================================================================
# SECTION 5 — MODEL TRAINING
# =============================================================================

print("\n[5/7] Training machine learning models...")

# --- Feature Selection ---
FEATURES = [
    'AGE', 'GENDER', 'BMI', 'SBP', 'DBP', 'Smoking Status',
    'Waist Circumference', 'Arm Circumference', 'INDFMPIR',
    'BMI_Category', 'BP_Stage', 'Age_Group', 'WHtR', 'Pulse_Pressure',
    'Alcohol Use', 'Average Drinks', 'Household Size'
]
TARGET = 'High_Risk'

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"  Features used ({len(FEATURES)}): {FEATURES}")
print(f"  Target: {TARGET} | Classes: {y.value_counts().to_dict()}")

# --- Train/Test Split (80/20 stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Scaled DataFrames (for tree models we use unscaled)
X_train_sc_df = pd.DataFrame(X_train_sc, columns=FEATURES)
X_test_sc_df  = pd.DataFrame(X_test_sc,  columns=FEATURES)

# --- Define Models ---
MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=15, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    'SVM':                 SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=11, metric='euclidean'),
}

# Tree models use raw features; others use scaled
NEEDS_SCALING = {'Logistic Regression', 'SVM', 'KNN'}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Model':<22} {'Accuracy':>9} {'F1':>9} {'AUC-ROC':>9} {'CV Mean':>9} {'CV Std':>8}")
print("  " + "-" * 68)

for name, model in MODELS.items():
    # Select scaled or unscaled
    Xtr = X_train_sc_df if name in NEEDS_SCALING else X_train
    Xte = X_test_sc_df  if name in NEEDS_SCALING else X_test

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': acc, 'f1': f1, 'auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'scaled': name in NEEDS_SCALING
    }
    print(f"  {name:<22} {acc:>8.4f} {f1:>9.4f} {auc:>9.4f} {cv_scores.mean():>9.4f} ±{cv_scores.std():.4f}")

# --- Best Model ---
# If several models tie by AUC, keep Gradient Boosting as the deployment default
# because it is usually smoother and more robust than a single decision tree.
max_auc = max(res['auc'] for res in results.values())
auc_tolerance = 1e-6
if abs(results['Gradient Boosting']['auc'] - max_auc) <= auc_tolerance:
    best_name = 'Gradient Boosting'
else:
    best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n   Best Model by AUC-ROC: {best_name} (AUC = {best['auc']:.4f})")

print(f"\n  --- Detailed Report: {best_name} ---")
print(classification_report(y_test, best['y_pred'], target_names=['Low Risk', 'High Risk']))

# =============================================================================
# SECTION 6 — MODEL EVALUATION VISUALIZATIONS
# =============================================================================

print("\n[6/7] Generating model evaluation plots...")

# --- FIGURE 2: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f172a')
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', color='white')

model_names = list(results.keys())
short_names = ['LR', 'DT', 'RF', 'GB', 'SVM', 'KNN']
metrics = {
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'F1 Score': [results[m]['f1']        for m in model_names],
    'AUC-ROC':  [results[m]['auc']       for m in model_names],
}

ax = axes[0]
ax.set_facecolor('#1e293b')
x = np.arange(len(model_names))
width = 0.25
for i, (metric, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + i*width, vals, width, label=metric, color=PALETTE[i], alpha=0.85, edgecolor='#0f172a')
ax.set_xticks(x + width); ax.set_xticklabels(short_names, color='#9ca3af')
ax.set_ylim(0.5, 1.0); ax.set_title('Accuracy / F1 / AUC per Model', color='white', fontweight='bold')
ax.legend(fontsize=9); ax.tick_params(colors='#9ca3af')
ax.set_facecolor('#1e293b')

# CV scores with error bars
ax = axes[1]
ax.set_facecolor('#1e293b')
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds  = [results[m]['cv_std']  for m in model_names]
ax.barh(short_names, cv_means, xerr=cv_stds, color=PALETTE[:len(model_names)],
        alpha=0.85, edgecolor='#0f172a', capsize=5, ecolor='white')
ax.set_xlim(0.5, 1.0)
ax.set_title('5-Fold Cross-Validation Accuracy', color='white', fontweight='bold')
ax.tick_params(colors='#9ca3af')
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=9, color='white')

# ROC Curves
ax = axes[2]
ax.set_facecolor('#1e293b')
for i, (name, res) in enumerate(results.items()):
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f"{short_names[i]} ({res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4)
ax.set_title('ROC Curves (All Models)', color='white', fontweight='bold')
ax.set_xlabel('False Positive Rate', color='#9ca3af')
ax.set_ylabel('True Positive Rate', color='#9ca3af')
ax.legend(fontsize=8, loc='lower right'); ax.tick_params(colors='#9ca3af')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig2_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  Saved: fig2_model_comparison.png")

# --- FIGURE 3: Best Model Deep-Dive ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f172a')
fig.suptitle(f'Best Model Deep-Dive: {best_name}', fontsize=14, fontweight='bold', color='white')

# Confusion Matrix
ax = axes[0]
ax.set_facecolor('#1e293b')
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            cbar=False, linewidths=0.5, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Confusion Matrix', color='white', fontweight='bold')
ax.set_xlabel('Predicted', color='#9ca3af')
ax.set_ylabel('Actual', color='#9ca3af')
ax.tick_params(colors='#9ca3af')

# Precision-Recall Curve
ax = axes[1]
ax.set_facecolor('#1e293b')
if best['y_proba'] is not None:
    prec, rec, _ = precision_recall_curve(y_test, best['y_proba'])
    ax.plot(rec, prec, color='#06b6d4', lw=2)
    ax.fill_between(rec, prec, alpha=0.15, color='#06b6d4')
ax.set_title('Precision-Recall Curve', color='white', fontweight='bold')
ax.set_xlabel('Recall', color='#9ca3af'); ax.set_ylabel('Precision', color='#9ca3af')
ax.tick_params(colors='#9ca3af')

# Feature Importance
ax = axes[2]
ax.set_facecolor('#1e293b')
best_model = best['model']
Xte_fi = X_test_sc_df if best['scaled'] else X_test

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    perm = permutation_importance(best_model, Xte_fi, y_test, n_repeats=10, random_state=42)
    importances = perm.importances_mean

feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True).tail(12)
colors_fi = ['#06b6d4' if v < feat_imp.max()*0.5 else '#ef4444' for v in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='#0f172a')
ax.set_title('Feature Importance (Top 12)', color='white', fontweight='bold')
ax.set_xlabel('Importance Score', color='#9ca3af')
ax.tick_params(colors='#9ca3af', labelsize=9)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig3_best_model_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  Saved: fig3_best_model_analysis.png")

# --- FIGURE 4: Decision Tree Rules (visual) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0f172a')
fig.suptitle('Decision Tree — Rules & Random Forest Depth Analysis', fontsize=14, fontweight='bold', color='white')

# DT text rules
ax = axes[0]
ax.set_facecolor('#1e293b')
dt_model = results['Decision Tree']['model']
dt_rules = export_text(dt_model, feature_names=FEATURES, max_depth=4)
ax.text(0.01, 0.99, dt_rules, transform=ax.transAxes, fontsize=6.5,
        verticalalignment='top', color='#a5f3fc',
        fontfamily='monospace', wrap=True)
ax.axis('off')
ax.set_title('Decision Tree Rules (depth ≤ 4)', color='white', fontweight='bold')

# RF: accuracy vs n_estimators
ax = axes[1]
ax.set_facecolor('#1e293b')
n_trees_range = [10, 20, 50, 75, 100, 150, 200]
rf_accs = []
for n in n_trees_range:
    rf_temp = RandomForestClassifier(n_estimators=n, max_depth=8, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    rf_accs.append(accuracy_score(y_test, rf_temp.predict(X_test)))
ax.plot(n_trees_range, rf_accs, 'o-', color='#34d399', lw=2, markersize=7)
ax.fill_between(n_trees_range, [a - 0.005 for a in rf_accs], [a + 0.005 for a in rf_accs],
                color='#34d399', alpha=0.15)
ax.set_title('Random Forest: Accuracy vs n_estimators', color='white', fontweight='bold')
ax.set_xlabel('Number of Trees', color='#9ca3af')
ax.set_ylabel('Test Accuracy', color='#9ca3af')
ax.tick_params(colors='#9ca3af')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig4_tree_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  Saved: fig4_tree_analysis.png")

# =============================================================================
# SECTION 7 — RISK PREDICTION FUNCTION (for integration with frontend)
# =============================================================================

print("\n[7/7] Setting up prediction API function...")

# Save best model pipeline info
best_model_obj = results[best_name]['model']

def predict_risk(age, gender, bmi, sbp, dbp, smoking_status,
                 waist_circ=95, arm_circ=32, income_pir=2.5,
                 alcohol_use=1, avg_drinks=1, household_size=3):
    """
    Predict lifestyle disease risk for a new individual.

    Parameters
    ----------
    age           : int   — Age in years
    gender        : int   — 1 = Male, 2 = Female
    bmi           : float — Body Mass Index (kg/m²)
    sbp           : float — Systolic blood pressure (mmHg)
    dbp           : float — Diastolic blood pressure (mmHg)
    smoking_status: int   — 1 = Non-smoker, 2 = Smoker
    waist_circ    : float — Waist circumference (cm)
    arm_circ      : float — Arm circumference (cm)
    income_pir    : float — Poverty Income Ratio
    alcohol_use   : int   — 1 = Drinks, 2 = Doesn't
    avg_drinks    : float — Average drinks per session
    household_size: int   — Number in household

    Returns
    -------
    dict with keys: risk_label, probability, risk_score, contributing_factors
    """
    # Engineer features
    bmi_cat = int(pd.cut([bmi], bins=[0,18.5,25,30,35,100], labels=[0,1,2,3,4])[0])
    bp_st   = bp_stage({'SBP': sbp, 'DBP': dbp})
    age_grp = int(pd.cut([age], bins=[0,30,45,60,200], labels=[0,1,2,3])[0])
    wh_ratio = waist_circ / 170  # assume avg height if not provided
    pulse_p  = sbp - dbp

    row = pd.DataFrame([[age, gender, bmi, sbp, dbp, smoking_status,
                         waist_circ, arm_circ, income_pir, bmi_cat, bp_st,
                         age_grp, wh_ratio, pulse_p, alcohol_use, avg_drinks, household_size]],
                       columns=FEATURES)

    # Scale if needed
    if best['scaled']:
        row_input = pd.DataFrame(scaler.transform(row), columns=FEATURES)
    else:
        row_input = row

    prob  = best_model_obj.predict_proba(row_input)[0][1]
    label = "High Risk" if prob >= 0.5 else "Low Risk"

    factors = []
    if sbp >= 130 or dbp >= 80: factors.append("Hypertension")
    if bmi >= 30:                factors.append("Obesity")
    if smoking_status == 2:      factors.append("Smoker")
    if age >= 45:                factors.append("Age ≥ 45")
    if wh_ratio > 0.5:          factors.append("High Waist-Height Ratio")

    return {
        "risk_label":          label,
        "probability":         round(float(prob), 4),
        "risk_pct":            round(float(prob) * 100, 1),
        "contributing_factors": factors,
        "model_used":          best_name,
    }

# --- Example Predictions ---
print("\n  --- Sample Predictions ---")
examples = [
    dict(age=62, gender=1, bmi=33, sbp=145, dbp=92, smoking_status=2, waist_circ=110),
    dict(age=28, gender=2, bmi=22, sbp=112, dbp=70, smoking_status=1, waist_circ=76),
    dict(age=48, gender=1, bmi=27, sbp=128, dbp=82, smoking_status=1, waist_circ=97),
]
for i, ex in enumerate(examples, 1):
    result = predict_risk(**ex)
    print(f"  Patient {i}: Age={ex['age']}, BMI={ex['bmi']}, SBP={ex['sbp']}")
    print(f"    → {result['risk_label']} | P={result['probability']} | Factors: {result['contributing_factors']}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"\n  {'Model':<22} {'Accuracy':>9} {'F1':>9} {'AUC-ROC':>9}")
print("  " + "-" * 52)
for name in model_names:
    r = results[name]
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<22} {r['accuracy']:>8.4f} {r['f1']:>9.4f} {r['auc']:>9.4f}{marker}")

print(f"""
  Key Findings:
  • Best Model         : {best_name} (AUC = {best['auc']:.4f})
  • Dataset Size       : {len(df)} samples after preprocessing
  • Features Used      : {len(FEATURES)} (including 5 engineered features)
  • Train/Test Split   : 80% / 20% (stratified)
  • Cross-Validation   : 5-Fold Stratified KFold
  • High-Risk Prevalence: {df['High_Risk'].mean()*100:.1f}% of dataset

  Output Files (in ./outputs/):
  • fig1_eda_overview.png
  • fig2_model_comparison.png
  • fig3_best_model_analysis.png
  • fig4_tree_analysis.png
""")
print("=" * 70)
print("  Pipeline complete.")
print("=" * 70)

# =============================================================================
# SAVE BEST MODEL FOR WEB APP INTEGRATION
# =============================================================================

import joblib
import json

model_slug_map = {
    'Logistic Regression': 'logistic_regression',
    'Decision Tree': 'decision_tree',
    'Random Forest': 'random_forest',
    'Gradient Boosting': 'gradient_boosting',
    'SVM': 'svm',
    'KNN': 'knn',
}

model_filename = 'lifestyle_risk_gb_model.pkl'
scaler_filename = 'lifestyle_risk_scaler.pkl'
config_filename = 'lifestyle_risk_config.json'

model_registry = {}
for model_name, result in results.items():
    slug = model_slug_map[model_name]
    filename = f'lifestyle_risk_{slug}_model.pkl'
    joblib.dump(result['model'], filename)
    model_registry[slug] = {
        'name': model_name,
        'file': filename,
        'needs_scaling': result['scaled'],
        'accuracy': float(result['accuracy']),
        'f1_score': float(result['f1']),
        'auc_roc': float(result['auc']),
    }

joblib.dump(best_model_obj, model_filename)
joblib.dump(scaler, scaler_filename)

# Save configuration for web app
config = {
    'model_name': best_name,
    'best_model_key': model_slug_map[best_name],
    'auc_roc': float(best['auc']),
    'accuracy': float(best['accuracy']),
    'f1_score': float(best['f1']),
    'features': FEATURES,
    'needs_scaling': best['scaled'],
    'model_type': model_slug_map[best_name],
    'models': model_registry,
}
with open(config_filename, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n Best model saved: {model_filename}")
print(f"   Model type: {best_name}")
print(f"   AUC-ROC: {best['auc']:.4f}")
print(f" Scaler saved: {scaler_filename}")
print(f" Config saved: {config_filename}")
