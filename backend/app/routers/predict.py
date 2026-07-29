from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.services.ensemble_predictor import EnsemblePredictor

router = APIRouter(
    prefix="/predict",
    tags=["Predict"],
)

predictor = EnsemblePredictor(
    efficientnet_checkpoint="model/unet_epoch8.pt",
    resnext_checkpoint="model/unet_epoch11.pt",
)


@router.post("")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)

        result = predictor.predict(image)
        mask_base64 = predictor._mask_to_base64(result["mask"])

        return {
            "gleason_score": result["gleason_score"],
            "mask": mask_base64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
