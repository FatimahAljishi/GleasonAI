from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from app.services.ensemble_predictor import EnsemblePredictor
from huggingface_hub import hf_hub_download

router = APIRouter(
    prefix="/predict",
    tags=["Predict"],
)

CACHE_DIR = "models"

_predictor = None


def get_predictor():
    global _predictor

    if _predictor is None:
        print("Loading models...")

        onnx_model = hf_hub_download(
            repo_id="FatimahAljishi/gleasonai-models",
            filename="unet_epoch8.onnx",
            local_dir=CACHE_DIR,
        )

        hf_hub_download(
            repo_id="FatimahAljishi/gleasonai-models",
            filename="unet_epoch8.onnx.data",
            local_dir=CACHE_DIR,
        )

        _predictor = EnsemblePredictor(
            onnx_path=onnx_model,
        )

        print("Models loaded.")

    return _predictor


@router.post("")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)

        print("Original size:", image.size)
        print("Mode:", image.mode)

        predictor = get_predictor()
        result = predictor.predict(image)
        mask_base64 = predictor._mask_to_base64(result["mask"])

        return {
            "gleason_score": result["gleason_score"],
            "mask": mask_base64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
