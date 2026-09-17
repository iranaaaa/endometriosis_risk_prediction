from fastapi import FastAPI
from pydantic import BaseModel
import csv
import os

app = FastAPI()

# 📁 CSV FILE PATH (your required path)
CSV_FILE = r"C:\Users\hp\.gemini\antigravity\scratch\endometriosis_risk_prediction\patients data\patients.csv"

# Ensure folder exists
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Create CSV file with headers if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "name",
            "age",
            "pelvic_pain",
            "menstrual_pain",
            "fatigue",
            "irregular_periods"
        ])

# 📦 Data model (matches your UI inputs)
class Patient(BaseModel):
    name: str
    age: int
    pelvic_pain: int
    menstrual_pain: int
    fatigue: int
    irregular_periods: int

# 🔥 API endpoint
@app.post("/submit")
def save_patient(data: Patient):

    # Save to CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            data.name,
            data.age,
            data.pelvic_pain,
            data.menstrual_pain,
            data.fatigue,
            data.irregular_periods
        ])

    return {"message": "Patient data saved successfully"}