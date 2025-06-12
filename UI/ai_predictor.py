import joblib
import json
import numpy as np
import os
import pandas as pd
import re
import traceback
import unicodedata
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from openai import OpenAI

# ---------- OpenAI ----------

client = OpenAI(api_key="sk-proj-KCpZqM0gBy1CZu0be_CRB66dLDev-3s8VjOEKxVVBCTlisoI3tn_xapwxtCuYfSl8zPAaN9nyK"
                        "T3BlbkFJEtu7S3IKu2uOfugimRLoLJkcu1YXwSiTe6mRwgH_jWpiEcgnZqT7GL46aNe9GT8R2-pj_csoMA")

# ---------- Model ----------
MODEL_PATH = r"C:\Users\krzys\Desktop\MAMMA_MIA_PROJEKT\models\xgb_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)
model = joblib.load(MODEL_PATH)

# ---------- Kolumny ----------
CLINICAL_XLSX = r"C:\Users\krzys\Desktop\MAMMA_MIA_PROJEKT\clinical_and_imaging_info.xlsx"
drop_cols = [
    "patient_id", "dataset", "acquisition_times", "mastectomy_post_nac",
    "days_to_follow_up", "days_to_recurrence", "days_to_metastasis", "days_to_death",
]
df = pd.read_excel(CLINICAL_XLSX).drop(columns=drop_cols, errors="ignore")
df = df[df["pcr"].notna()]
EXPECTED_COLUMNS = df.drop(columns=["pcr"]).columns.tolist()

# ---------- FastAPI ----------
app = FastAPI()

# ---------- Dynamic prompt ----------
GPT_PROMPT = (
        "Extract the following fields from the patient's description below "
        "and return them as clean JSON (no comments):\n\n"
        + ", ".join(EXPECTED_COLUMNS)
        + "\n\nPatient text:\n"
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict_pcr")
async def predict_pcr(request: Request):
    # 1) wejście
    body = await request.json()
    raw_text = body.get("raw_text", "").strip()
    if not raw_text:
        raise HTTPException(400, "raw_text missing")

    raw_text = unicodedata.normalize("NFKD", raw_text).encode("ascii", "ignore").decode()

    # 2) GPT
    try:
        gpt = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                
    You are a medical assistant helping an oncologist analyze patient data before treatment.

    Your task is to extract structured, clean data from the patient's clinical description in English and return it as 
    valid JSON. The model you support uses this data to predict pathological complete response (pCR) to neoadjuvant 
    chemotherapy in breast cancer.
    
    Always return JSON with fields matching exactly those requested. Do not include comments or explanations — 
    just clean JSON.
    
    Interpret missing or vague data as `"unknown"`.
    
    Be precise. If the patient shows favorable indicators (e.g., HER2-positive, high Ki-67, triple-negative subtype), 
    this may indicate a higher probability of achieving pCR (label: **1**). If unfavorable (e.g., luminal A, low grade, 
    low proliferation), the chance of pCR is lower (label: **0**).
    
    Do not speculate — if the description is ambiguous, return `"unknown"` for uncertain fields.
    
    Your output is consumed by an XGBoost model, which expects consistency and structure.
    
    Never apologize or generate human-like conversation — your role is to assist clinical processing.

                """
                 },
                {"role": "user", "content": GPT_PROMPT + raw_text},
            ],
        )
        text_json = re.sub(r"```json|```", "", gpt.choices[0].message.content).strip()
        patient_dict = json.loads(text_json)
        df_in = pd.DataFrame([patient_dict])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"GPT/JSON error: {e}")

    # 3) dopasowanie kolumn
    missing = [c for c in EXPECTED_COLUMNS if c not in df_in.columns]
    for c in missing:
        df_in[c] = pd.NA
    df_in = df_in[EXPECTED_COLUMNS]

    # kategorie
    for c in df_in.columns:
        if df_in[c].dtype == "object":
            df_in[c] = df_in[c].astype("category")

    # 4) predykcja
    try:
        pred = int(model.predict(df_in)[0])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Model error: {e}")

    # 5) serializacja – usuwamy NaN / inf
    df_safe = (
        df_in.replace({pd.NA: "unknown", np.nan: "unknown",
                       np.inf: "unknown", -np.inf: "unknown"})
        .astype(str)
    )

    resp = {
        "prediction": pred,
        "model_used": "XGBoost",
        "extracted_data": df_safe.to_dict(orient="records")[0],
    }
    if missing:
        resp["warning"] = {
            "message": "Missing fields – prediction may be less accurate",
            "missing_fields": missing,
        }
    return jsonable_encoder(resp)
