from ocr.tool.translate import (
    build_model,
    translate,
    process_input,
)
from ocr.tool.utils import download_weights
from ocr.tool.config import Cfg

import torch
from collections import defaultdict


class Predictor:
    def __init__(self, config: Cfg):

        device = config["device"]

        model, vocab = build_model(config)
        weights = "/tmp/weights.pth"

        if config["weights"].startswith("http"):
            weights = download_weights(config["weights"])
        else:
            weights = config["weights"]

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device
        
    def predict(self, img, return_prob=False):
        img = process_input(
            img,
            self.config["dataset"]["image_height"],
            self.config["dataset"]["image_min_width"],
            self.config["dataset"]["image_max_width"],
        )
        img = img.to(self.config["device"])

        s, prob = translate(img, self.model)
        s = s[0].tolist()
        prob = prob[0]

        s = self.vocab.decode(s)

        if return_prob:
            return s, prob
        else:
            return s