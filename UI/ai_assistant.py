from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import openai
import json
import constants

# === Konfiguracja OpenAI API ===
# Ustaw swój klucz API tutaj (wklej z https://platform.openai.com/account/api-keys)
openai.api_key = "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"

# === Wczytanie modelu ===
MODEL_PATH = constants.models_folder_path + "\\xgb_model.pkl"
model = joblib.load(MODEL_PATH)

# === Załaduj kolumny oczekiwane przez model ===
CLINICAL_XLSX = constants.project_root_folder + "\\clinical_and_imaging_info.xlsx"
columns_to_drop = [
    'patient_id', 'dataset',
    'acquisition_times', 'mastectomy_post_nac', 'days_to_follow_up', 'days_to_recurrence',
    'days_to_metastasis', 'days_to_death',
]
df = pd.read_excel(CLINICAL_XLSX)
df = df.drop(columns=columns_to_drop, errors='ignore')
df = df[df["pcr"].notna()]
EXPECTED_COLUMNS = df.drop(columns=["pcr"]).columns.tolist()

# === FastAPI app ===
app = FastAPI()


# === Prompt dla GPT (po angielsku) ===
GPT_PROMPT = """
Extract clinical data from the patient's description below.
Return the result in clean JSON format, without comments or explanation.

Required fields:
- age: integer
- her2: 0 or 1
- er: 0 or 1
- pr: 0 or 1
- menopause: "pre" or "post"
- tumor_subtype: string
- therapy: list of strings (e.g. ["anthracycline", "taxane"])

Patient text:
"""

# === Endpoint do predykcji ===
@app.post("/predict_pcr")
def predict_pcr(self):
    input_data = self.user_input.get().strip()

    self.chat_display.insert("end", f"Ty: {input_data}\n")
    self.user_input.delete(0, "end")

    # 1. Wywołanie GPT-4
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant helping an oncologist."},
                {"role": "user", "content": GPT_PROMPT + input_data}
            ]
        )
        extracted_json = completion.choices[0].message['content']
        patient_dict = json.loads(extracted_json)
        patient_data = pd.DataFrame([patient_dict])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during GPT extraction or JSON parsing: {str(e)}")

    # 2. Dopasuj brakujące kolumny jako NaN (ale nie zwracaj błędu)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in patient_data.columns]
    for col in EXPECTED_COLUMNS:
        if col not in patient_data.columns:
            patient_data[col] = pd.NA
    patient_data = patient_data[EXPECTED_COLUMNS]

    # 3. Predykcja modelem
    try:
        prediction = model.predict(patient_data)[0]
        response = {
            "prediction": int(prediction),
            "model_used": "XGBoost",
            "extracted_data": patient_data.to_dict(orient="records")[0]
        }

        # Dodaj komunikat o brakujących danych tylko jeśli jakieś były
        if missing_cols:
            response["warning"] = {
                "message": "Some clinical information was missing. Prediction may be less accurate.",
                "missing_fields": missing_cols
            }
        self.chat_display.insert("end", f"Asystent: {response['extracted_data']}\n\n")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")