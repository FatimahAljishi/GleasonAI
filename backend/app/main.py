from fastapi import FastAPI, UploadFile
from PIL import Image
from app.services.ensemble_predictor import EnsemblePredictor
from fastapi.middleware.cors import CORSMiddleware
from app.routers.predict import router as predict_router

app = FastAPI(
    title="GleasonAI API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://gleason-ai.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "GleasonAI API is running"}


app.include_router(predict_router)
