
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("xgboost_model.joblib")

class InputData(BaseModel):
    age_ben_snds: int
    asu_nat: int
    ben_qlt_cod: int
    ben_sex_cod: int
    ben_res_reg: int
    flt_rem_mnt: float
    flt_act_qte: int
    prs_nat: int

@app.get("/")
def root():
    return {"message": "C2S prediction API is running."}

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[
        data.age_ben_snds,
        data.asu_nat,
        data.ben_qlt_cod,
        data.ben_sex_cod,
        data.ben_res_reg,
        data.flt_rem_mnt,
        data.flt_act_qte,
        data.prs_nat
    ]])
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]
    return {"prediction": int(prediction), "probability": round(float(proba), 4)}
