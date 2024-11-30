import os
import time
import cv2
from dotenv import load_dotenv
import numpy as np
import argparse
import torch
import gdown
from torch.autograd import Variable

from src.utils import craft_utils
from src.utils import image_utils
from src.utils import file_utils

from src.craft_model.craft import CRAFT

load_dotenv()

# Download craft_weight
# url_weights_craft_25k = os.getenv('URL_WEIGHTS_CRAFT_25K')
# weights_folder = os.getenv('WEIGHTS_FOLDER')
# if not os.path.isdir(weights_folder):
#     os.mkdir(weights_folder)
# gdown.download(url_weights_craft_25k, weights_folder, fuzzy=True, quiet=False)


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default=os.getenv('TRAINED_MODEL'), type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=float(os.getenv('TEXT_THRESHOLD')), type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=float(os.getenv('LOW_TEXT')), type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=float(os.getenv('LINK_THRESHOLD')), type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=int(os.getenv('CANVAS_SIZE')), type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=float(os.getenv('MAG_RATIO')), type=float, help='image magnification ratio')
parser.add_argument('--poly', default=os.getenv('POLY').lower() == 'true', action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=os.getenv('SHOW_TIME').lower() == 'true', action='store_true', help='show processing time')
parser.add_argument('--input_root_folder', default=os.getenv('INPUT_ROOT_FOLDER'), type=str, help='folder path to input images')

args = parser.parse_args()

image_list, _, _ = file_utils.get_files(args.input_root_folder)

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

        output_path = f"{output_folder}/cropped_{idx}.png"
        cv2.imwrite(output_path, cropped_image)
        print(f"Save image: {output_path}")

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
        

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = output_craft_detect_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=output_craft_detect_folder)

        # save region images
        extract_text_regions(image_path, bounding_boxes, output_cutting_detect_folder)

    print("elapsed time : {}s".format(time.time() - start_time))
