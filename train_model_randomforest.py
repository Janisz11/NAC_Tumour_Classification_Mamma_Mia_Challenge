import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing_pipeline import get_pipeline
import joblib
import csv
import os


clinical_info_path = '/lustre/pd03/hpc-ljelen-1692966897/mamma_mia/clinical_and_imaging_info.xlsx'


clinical_data = pd.read_excel(clinical_info_path)
df = clinical_data

#  Usunięcie wierszy, gdzie pcr jest null
df = df[df["pcr"].notna()]

# 2. Podział na cechy i etykiety
X = df.drop(columns=["pcr"])
y = df["pcr"].astype(int)

# 3. Pipeline + transformacja
pipeline = get_pipeline()
X_transformed = pipeline.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 5. Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Predykcja i ewaluacja
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# 7. Zapis logów
csv_file = "training_log.csv"
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


