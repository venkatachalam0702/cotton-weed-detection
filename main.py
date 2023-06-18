# Import FastAPI
import torch
import json
from PIL import Image
from fastapi import File, FastAPI, Form
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/getapi')
def getapi():
    return {"message": "GET API test"}


@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
        input_image = Image.open(io.BytesIO(file)).convert("RGB")
        model=torch.hub.load('ultralytics/yolov5', 'custom', path="sbest.pt", force_reload=True)
        results = model(input_image)
        results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
        print(results_json)
        return results_json