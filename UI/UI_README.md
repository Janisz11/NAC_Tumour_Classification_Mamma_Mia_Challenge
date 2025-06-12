# Breast Cancer AI Interface

A GUI-based assistant for analyzing breast cancer patient data, visualizing MRI scans, and predicting pathological complete response (pCR) using a machine learning model.

## Features

- Load and visualize `.nii.gz` MRI scan files.
- Navigate through patient scan slices.
- Browse patient data from Excel sheets.
- Chat with an AI assistant for help or pCR prediction.
- Automatic data extraction from text using OpenAI GPT.
- Backend FastAPI server handles predictions via a trained XGBoost model.

---

## 🔧 Installation

1. **Install dependencies** (Python ≥ 3.8):

```bash
pip install -r requirements.txt
```
Minimal list:
```text
fastapi
uvicorn
openai
pandas
numpy
joblib
nibabel
ttkbootstrap
matplotlib
scikit-learn
python-multipart
```
## 🚀 Usage
Start backend API (FastAPI server):

```bash
uvicorn ai_predictor:app --reload
```
Run the interface:
```bash
python main.py
```

## 🧠 Example AI Query
Paste the following in the assistant:

```text
I have a patient that is 38 years old. Her HER2 is 1, ER is 1, and PR is 0. She is premenopausal. 
The tumor subtype is HER2-enriched. She received neoadjuvant therapy with HER2-targeted drugs. 
She's hispanic and does not have breast implants.
The assistant will extract the structured data and send it to the predictor.
```

## 🗂️ Notes
- Patient data is read from clinical_and_imaging_info.xlsx.

- MRI images must be stored under images/<patient_id>/*.nii.gz.

- Model expects preprocessed clinical data with specific column names.

- constants.py should contain absolute paths like:

```python
clinical_and_imaging_info_path = r"...path...\clinical_and_imaging_info.xlsx"
images_folder_path = r"...path...\images"
```