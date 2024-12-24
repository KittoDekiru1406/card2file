from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from ocr.tool.predictor import Predictor
from ocr.tool.config import Cfg
from PIL import Image, UnidentifiedImageError
import data_processing
import io
import os


# Initialize FastAPI app
app = FastAPI()


# Lazy load the model using a singleton pattern

def get_detector() -> Predictor:
    if not hasattr(app.state, "detector"):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        app.state.detector = Predictor(config)
    return app.state.detector

def get_file_paths_from_folder(folder: str) -> list:
    file_paths = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            file_paths.append(file_path)
    file_paths = sorted(file_paths)
    return file_paths

# ======================== OCR Functions ========================
def ocr_predict(image: Image) -> str:
    detector = get_detector()
    text = detector.predict(image)
    return text 

def ocr_batch():
    results = data_processing.main()
    return results

def ocr_extract(folder: str) -> dict:
    pass

# ======================== API Endpoints ========================   
class OCRPredictResponse(BaseModel):
    text: str

class OCRBatchResponse(BaseModel):
    results: dict

# ======================== API Endpoints ========================

@app.post("/api/ocr/predict", response_model=OCRPredictResponse)
async def api_ocr_predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid file type. Please upload an image.")

        # Read image content
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
        except UnidentifiedImageError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Uploaded file is not a valid image.")

        # Predict text
        text = ocr_predict(image)

        # Return response
        return OCRPredictResponse(text=text)

    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the internal error
        print(f"Internal error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred while processing the image.")
        

@app.post("/api/ocr/batch")
async def api_ocr_batch():
    
    results = ocr_batch()
    return results

@app.post("/api/ocr/extract")
async def api_ocr_extract(folder: str):
    pass