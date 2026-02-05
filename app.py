from fastapi import FastAPI, UploadFile, File
import torch
from predict_model import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Grass Growth Prediction API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return {"prediction": result}
