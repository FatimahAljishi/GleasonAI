from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from app.services.ensemble_predictor import EnsemblePredictor
from huggingface_hub import hf_hub_download

router = APIRouter(
    prefix="/predict",
    tags=["Predict"],
)

CACHE_DIR = "models"

efficientnet_checkpoint = hf_hub_download(
    repo_id="FatimahAljishi/gleasonai-models",
    filename="unet_epoch8.pt",
    local_dir=CACHE_DIR,
)

resnext_checkpoint = hf_hub_download(
    repo_id="FatimahAljishi/gleasonai-models",
    filename="unet_epoch11.pt",
    local_dir=CACHE_DIR,
)

predictor = EnsemblePredictor(
    efficientnet_checkpoint=efficientnet_checkpoint,
    resnext_checkpoint=resnext_checkpoint,
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
