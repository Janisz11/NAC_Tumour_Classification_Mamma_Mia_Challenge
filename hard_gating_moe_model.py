import pandas as pd
import numpy as np
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

# Load data
clinical_info_path = 'data_workspace/gtp_NCCN_based_filled_data_clinical_info.xlsx'
df = pd.read_excel(clinical_info_path)
df = df[df["pcr"].notna()]
X = df.drop(columns=["pcr"])
y = df["pcr"].astype(int)

# Data preprocessing
pipeline = get_pipeline()
X_transformed = pipeline.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

# Expert 1: XGBoost
xgb = XGBClassifier(
    n_estimators=124,
    max_depth=3,
    learning_rate=0.0411,
    subsample=0.978,
    colsample_bytree=0.623,
    gamma=0.769,
    reg_alpha=0.749,
    reg_lambda=0.781,
    eval_metric='error',
    use_label_encoder=False,
    tree_method='hist',
    random_state=42
)
xgb.fit(X_train, y_train)

# Expert 2: RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create gating labels: 0 = XGBoost correct, 1 = RF correct, else 0
xgb_preds = xgb.predict(X_train)
rf_preds = rf.predict(X_train)

xgb_correct = (xgb_preds == y_train.values)
rf_correct = (rf_preds == y_train.values)

gating_labels = ((rf_correct) & (~xgb_correct)).astype(int)

# Gating model with best Optuna parameters (Trial 11)
gating_model = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_error',
    boosting_type='gbdt',
    verbosity=-1,
    n_estimators=200,
    learning_rate=0.08225826152779128,
    max_depth=2,
    num_leaves=62,
    subsample=0.8529811277498729,
    colsample_bytree=0.9221309236747597,
    reg_alpha=0.09452586401207042,
    reg_lambda=1.432580643025585
)
gating_model.fit(X_train, gating_labels)

# Test predictions from experts
xgb_test_preds = xgb.predict(X_test)
rf_test_preds = rf.predict(X_test)

# Gating model predictions
gating_choices = gating_model.predict(X_test)  # 0 = XGB, 1 = RF

# Count expert selections
num_rf = np.sum(gating_choices == 1)
num_xgb = np.sum(gating_choices == 0)

print(f"Gating selected RandomForest: {num_rf} times")
print(f"Gating selected XGBoost: {num_xgb} times")

# Final predictions based on gating decision
final_preds = np.where(gating_choices == 0, xgb_test_preds, rf_test_preds)

# Evaluation
report = classification_report(y_test, final_preds, output_dict=True)
accuracy = accuracy_score(y_test, final_preds)

# Save results
csv_file = "hard_gating_fixed_params_log.csv"
fieldnames = ["accuracy"] + [
    f"{label}_{metric}"
    for label in report
    if isinstance(report[label], dict)
    for metric in report[label]
]

row = {"accuracy": accuracy}
for label in report:
    if isinstance(report[label], dict):
        for metric in report[label]:
            row[f"{label}_{metric}"] = report[label][metric]

file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

# Display results
print("Hard Gating accuracy (fixed Optuna params):", round(accuracy, 4))
