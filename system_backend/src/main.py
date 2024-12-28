from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from pydantic import BaseModel
from detect_infor.run_craft import predict
from data_processing import post_processing
from ocr.run_ocr import ocr_folder_to_csv
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import os
from dotenv import load_dotenv

load_dotenv()

root_image_dir = os.getenv('INPUT_ROOT_FOLDER')
cutting_image_folder_path = os.getenv('OUTPUT_CUTTING_DETECT_FOLDER')
craft_detect_folder = os.getenv('OUTPUT_CRAFT_DETECT_FOLDER')

# Initialize FastAPI app
app = FastAPI(title="OCR APP")


# ======================== API Endpoints ========================   
class OCRResponse(BaseModel):
    results: dict

# ======================== API Endpoints ========================

@app.post("/api/ocr", response_model=OCRResponse)
async def predictocr(file: UploadFile = File(...), filename: str = Form(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid file type. Please upload an image.")

        # Read image content
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))

            save_path = os.path.join(root_image_dir, filename)
            image.save(save_path)

        except UnidentifiedImageError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Uploaded file is not a valid image.")
        image = np.array(image)
        predict(image, save_path)

        output_folder = './database/data_output'
        output_csv = os.path.join(output_folder, 'ocr_output.csv')
        os.makedirs(output_folder, exist_ok=True)

        output_csv = './database/data_output/ocr_output.csv'
        ocr_folder_to_csv(cutting_image_folder_path, output_csv)
        for _ in range(3):
            results = post_processing(output_csv, craft_detect_folder)
        
        return OCRResponse(results=results)


    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the internal error
        print(f"Internal error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred while processing the image.")
    
