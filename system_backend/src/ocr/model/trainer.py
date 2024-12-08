from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torch

from ocr.tool.config import Cfg
from ocr.loader.aug import ImgAugTransformV2
from ocr.tool.translate import build_model, translate
from ocr.tool.logger import Logger
from ocr.tool.utils import download_weights
from ocr.optim.labelsmoothingloss import LabelSmoothingLoss
from ocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator


import time
import numpy as np


class Trainer:
    def __init__(self, 
                 config: Cfg, 
                 pretrained: str=True,
                 augmentor: classmethod=ImgAugTransformV2):
        
        self.config = config
        self.model, self.vocab = build_model(config)
        self.device = config["device"]
        self.num_iters = config["trainer"]["iters"]
        self.beamsearch = config["predictor"]["beamsearch"]

        self.data_root = config["dataset"]["data_root"]
        self.train_annotation = config["dataset"]["train_annotation"]
        self.valid_annotation = config["dataset"]["valid_annotation"]
        self.dataset_name = config["dataset"]["name"]

        self.batch_size = config["trainer"]["batch_size"]
        self.print_every = config["trainer"]["print_every"]
        self.valid_every = config["trainer"]["valid_every"]

        self.image_aug = config["aug"]["image_aug"]
        self.masked_language_model = config["aug"]["masked_language_model"]

        self.checkpoint = config["trainer"]["checkpoint"]
        self.export_weights = config["trainer"]["export"]
        self.metrics = config["trainer"]["metrics"]
        logger = config["trainer"]["log"]
        
        
        if logger is None:
            self.logger = Logger(logger)
            
        if pretrained:
            weight_file = download_weights(config["pretrain"], quiet=config["quiet"])
            self.load_weights(weight_file)
            
        self.iter = 0
        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(
            self.optimizer, total_steps=self.num_iters, **config["optimizer"]
        )
        self.criterion = LabelSmoothingLoss(
            len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1
        )
        
        transforms = None
        if self.image_aug:
            transforms = augmentor
            
            self.train_gen = self.data_gen(
            "train_{}".format(self.dataset_name),
            self.data_root,
            self.train_annotation,
            self.masked_language_model,
            transform=transforms,
        )
        if self.valid_annotation:
            self.valid_gen = self.data_gen(
                "valid_{}".format(self.dataset_name),
                self.data_root,
                self.valid_annotation,
                masked_language_model=False,
            )

        self.train_losses = []
        
    def train(self):
        total_loss = 0

        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = "iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}".format(
                    self.iter,
                    total_loss / self.print_every,
                    self.optimizer.param_groups[0]["lr"],
                    total_loader_time,
                    total_gpu_time,
                )

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info)
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision(self.metrics)

                info = "iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}".format(
                    self.iter, val_loss, acc_full_seq, acc_per_char
                )
                print(info)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights)
                    best_acc = acc_full_seq
                                 
    def validate(self):
        self.model.eval()

        total_loss = []

        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = (
                    batch["img"],
                    batch["tgt_input"],
                    batch["tgt_output"],
                    batch["tgt_padding_mask"],
                )

                outputs = self.model(img, tgt_input, tgt_padding_mask)
                #                loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

                outputs = outputs.flatten(0, 1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())

                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()

        return total_loss

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []

        for batch in self.valid_gen:
            batch = self.batch_to_device(batch)


            translated_sentence, prob = translate(batch["img"], self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch["tgt_output"].tolist())

            img_files.extend(batch["filenames"])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)

            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob
    
    def data_gen(
        self,
        lmdb_path,
        data_root,
        annotation,
        masked_language_model=True,
        transform=None,
    ):
        dataset = OCRDataset(
            lmdb_path=lmdb_path,
            root_dir=data_root,
            annotation_path=annotation,
            vocab=self.vocab,
            transform=transform,
            image_height=self.config["dataset"]["image_height"],
            image_min_width=self.config["dataset"]["image_min_width"],
            image_max_width=self.config["dataset"]["image_max_width"],
        )

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            **self.config["dataloader"]
        )

        return gen