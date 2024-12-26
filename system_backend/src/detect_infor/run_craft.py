import os
import time
import cv2
from dotenv import load_dotenv
import numpy as np
import argparse
import torch
from torch.autograd import Variable
import subprocess

from src.utils import craft_utils
from src.utils import image_utils
from src.utils import file_utils

from src.craft_model.craft import CRAFT

load_dotenv()

# Download craft_weight
url_weights_craft_25k = os.getenv('URL_WEIGHTS_CRAFT_25K')
weights_folder = os.getenv('WEIGHTS_FOLDER')
if not os.path.isdir(weights_folder):
    os.mkdir(weights_folder)
output_path = os.getenv('TRAINED_MODEL')

if not os.path.isfile(output_path):
    print("File chưa tồn tại, tiến hành tải về...")
    subprocess.run(["wget", "--no-check-certificate", url_weights_craft_25k, "-O", output_path])
    print(f"File đã được tải về: {output_path}")
else:
    print(f"File đã tồn tại tại: {output_path}, bỏ qua quá trình tải.")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default=os.getenv('TRAINED_MODEL'), type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=float(os.getenv('TEXT_THRESHOLD')), type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=float(os.getenv('LOW_TEXT')), type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=float(os.getenv('LINK_THRESHOLD')), type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=int(os.getenv('CANVAS_SIZE')), type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=float(os.getenv('MAG_RATIO')), type=float, help='image magnification ratio')
parser.add_argument('--poly', default=os.getenv('POLY').lower() == 'true', action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=os.getenv('SHOW_TIME').lower() == 'true', action='store_true', help='show processing time')
parser.add_argument('--output_after_preprocessing_folder', default=os.getenv('OUTPUT_AFTER_PREPROCESSING_FOLDER'), type=str, help='folder path to input images')

args = parser.parse_args()

image_list, _, _ = file_utils.get_files(args.output_after_preprocessing_folder)

# Create output folder
output_craft_detect_folder = os.getenv('OUTPUT_CRAFT_DETECT_FOLDER')
if not os.path.isdir(output_craft_detect_folder):
    os.mkdir(output_craft_detect_folder)

output_cutting_detect_folder = os.getenv('OUTPUT_CUTTING_DETECT_FOLDER')
if not os.path.isdir(output_cutting_detect_folder):
    os.mkdir(output_cutting_detect_folder)

def run_net(net, image, text_threshold, link_threshold, low_text, poly):
    start_time = time.time()

    #resize image
    img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = image_utils.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)        # [h, w, c] => [c, h, w]
    x = Variable(x.unsqueeze(0))                    # [c, h, w] => [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    
    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    end_time_1 = time.time() - start_time
    start_time = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range((len(polys))):
        if polys[k] is None:
            polys[k] = boxes[k]

    end_time_2 = time.time() - start_time

    # render results 
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = image_utils.cvt2HeatmapImg(render_img)

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(end_time_1, end_time_2))

    return boxes, polys, ret_score_text

def are_in_same_line(box1, box2, vertical_threshold):
    y1_top = min(box1[1], box1[3])
    y2_top = min(box2[1], box2[3])
    y1_bottom = max(box1[5], box1[7])
    y2_bottom = max(box2[5], box2[7])
    return abs(y1_top - y2_top) <= vertical_threshold or abs(y1_bottom - y2_bottom) <= vertical_threshold


def group_bounding_boxes(bounding_boxes, horizontal_threshold, vertical_threshold):
    grouped_boxes = []
    visited = set()

    for i, box1 in enumerate(bounding_boxes):
        if i in visited:
            continue
        group = [box1]
        visited.add(i)

        for j, box2 in enumerate(bounding_boxes):
            if j in visited:
                continue
            if are_in_same_line(box1, box2, vertical_threshold) and \
               abs(box1[2] - box2[0]) <= horizontal_threshold:
                group.append(box2)
                visited.add(j)

        # Cập nhật tọa độ box sau khi gộp
        x_coords = [coord for box in group for coord in box[::2]]
        y_coords = [coord for box in group for coord in box[1::2]]
        grouped_boxes.append([
            min(x_coords), min(y_coords),
            max(x_coords), min(y_coords),
            max(x_coords), max(y_coords),
            min(x_coords), max(y_coords)
        ])

    return np.array(grouped_boxes)



def extract_text_regions(image_path: str, bounding_boxes, output_folder="output"):
    """
        This function to extract the text regions on bouding boxes and save that images

        Args:
            image_path (str): image path
            bounding_boxes (list): bounding boxes list
            output_folder (str): folder to save image

        Returns:
            None
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Read root image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    
    # Execute
    for idx, box in enumerate(bounding_boxes):
        
        points = np.array([
            [box[0], box[1]], 
            [box[2], box[3]], 
            [box[4], box[5]], 
            [box[6], box[7]]
        ], dtype=np.int32)

        rect = cv2.boundingRect(points)
        x, y, w, h = rect

        cropped_image = image[y:y+h, x:x+w]
        
        if idx < 10:
            output_path = f"{output_folder}/0{idx}.png"
        else:
            output_path = f"{output_folder}/{idx}.png"
    
        cv2.imwrite(output_path, cropped_image)
        print(f"Save image: {output_path}")


def predict(image: np.ndarray) -> None:
    '''
        This function is to predict a image

        Args:
            image: a numpy array

        Returns:
            None
    '''
    net = CRAFT()
    print('Loading weights from checkpoint (' + args.trained_model + ')')

    net.load_state_dict(file_utils.copyStateDict(torch.load(args.trained_model, map_location=torch.device('cpu'))))

    net.eval()
    
    bboxes, polys, score_text = run_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.poly)

    # position of bouding boxes
    bounding_boxes = np.array(polys)
    bounding_boxes = np.squeeze(bounding_boxes)
    bounding_boxes = np.array([box.flatten() for box in bounding_boxes])
        
    # Group near bounding boxes
    bounding_boxes = group_bounding_boxes(bounding_boxes, horizontal_threshold=2000, vertical_threshold=15)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = output_craft_detect_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=output_craft_detect_folder)

    # save region images
    extract_text_regions(image_path, bounding_boxes, output_cutting_detect_folder)



if __name__ == '__main__':
    # load net
    net = CRAFT()

    print('Loading weights from checkpoint (' + args.trained_model + ')')

    net.load_state_dict(file_utils.copyStateDict(torch.load(args.trained_model, map_location=torch.device('cpu'))))

    net.eval()

    start_time = time.time()

    bounding_boxes = None

    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = image_utils.loadImage(image_path)

        bboxes, polys, score_text = run_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.poly)

        # position of bouding boxes
        bounding_boxes = np.array(polys)
        bounding_boxes = np.squeeze(bounding_boxes)
        bounding_boxes = np.array([box.flatten() for box in bounding_boxes])
         
        # Group near bounding boxes
        bounding_boxes = group_bounding_boxes(bounding_boxes, horizontal_threshold=2000, vertical_threshold=15)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = output_craft_detect_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=output_craft_detect_folder)

        # save region images
        extract_text_regions(image_path, bounding_boxes, output_cutting_detect_folder)

    print("elapsed time : {}s".format(time.time() - start_time))