from pathlib import Path
import joblib
import pandas as pd
MODEL_PATH=Path(__file__).resolve().parents[1]/"models"/"model_bundle.pkl"
def load_bundle(): return joblib.load(MODEL_PATH)
def predict_geek_profile(respuestas):
    bundle=load_bundle()
    values=list(respuestas.values()) if isinstance(respuestas,dict) else list(respuestas)
    if len(values)!=10: raise ValueError("Se requieren exactamente 10 respuestas.")
    X=pd.DataFrame([values],columns=bundle["features"]); model=bundle["model"]
    profile=model.predict(X)[0]; probs=model.predict_proba(X)[0]
    return profile,{c:float(p) for c,p in zip(model.classes_,probs)}
