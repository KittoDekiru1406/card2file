import os
import csv
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from ocr.tool.predictor import Predictor
from ocr.tool.config import Cfg
import numpy as np


load_dotenv()

def get_detector() -> Predictor:
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    return Predictor(config)

def ocr_image(detector: Predictor, image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            image = Image.open(f)
            text = detector.predict(image)
        return text
    except UnidentifiedImageError:
        return "Error: Invalid image format"
    except Exception as e:
        return f"Error: {str(e)}"

def ocr_folder_to_csv(folder_path: str, output_csv: str):
    if not os.path.isdir(folder_path):
        print("Đường dẫn thư mục không hợp lệ.")
        return

    detector = get_detector()

    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Path", "Extracted Text"])  

            for file_name in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    print(f"Đang xử lý: {file_path}")
                    text = ocr_image(detector, file_path)
                    writer.writerow([file_path, text])
        print(f"Kết quả đã được lưu vào: {output_csv}")
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")

def mainn():
    folder_path = os.getenv('OUTPUT_CUTTING_DETECT_FOLDER')
    output_csv = './database/data_output/ocr_output.csv'


    if not folder_path:
        print("Biến môi trường 'OUTPUT_CUTTING_DETECT_FOLDER' chưa được thiết lập.")
    elif not output_csv:
        print("Biến môi trường 'OUTPUT_OCR' chưa được thiết lập.")
    else:
        ocr_folder_to_csv(folder_path, output_csv)

