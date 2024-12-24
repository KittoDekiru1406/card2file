import os
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from ocr.tool.predictor import Predictor
from ocr.tool.config import Cfg

load_dotenv()

def get_detector() -> Predictor:
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    return Predictor(config)

def ocr_image(detector: Predictor, image_path: str) -> str:
    with open(image_path, "rb") as f:
            image = Image.open(f)
            text = detector.predict(image)
    return text

def process_images_in_folder(folder_path: str) -> pd.DataFrame:
    detector = get_detector()
    results = []

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            text = ocr_image(detector, file_path)
            results.append([file_path, text])

    df = pd.DataFrame(results, columns=["File Path", "Extracted Text"])
    return df
def data():
    folder_path = os.getenv('OUTPUT_CUTTING_DETECT_FOLDER')  
    data = process_images_in_folder(folder_path)
    return data
# if __name__ == "__main__":
#     folder_path = os.getenv('OUTPUT_CUTTING_DETECT_FOLDER')  
#     data = process_images_in_folder(folder_path)
    