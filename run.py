import json
import os
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import logging

import torch
import torch.nn as nn
import torchvision
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

logger = logging.getLogger("detectron2")
logger.setLevel(logging.CRITICAL)

TEST_IMAGES_PATH, SAVE_PATH = r'C:\Users\Egor\Desktop\NTO Final\train_segmentation\images', \
                              r'C:\Users\Egor\Desktop\NTO Final'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGM_MODEL_PATH = "segm-model_final.pth"
OCR_MODEL_PATH = "ocr-model-last.ckpt"

CONFIG_JSON = {
    "alphabet": r""" !"'()+,-./0123456789:;=?IN[]АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№""",
    # "alphabet": r"""' !"%\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz|}ЁАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№'""",
    "image": {"width": 256, "height": 100},
}


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


class SEGMpredictor:
    def __init__(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = 1000
        cfg.INPUT.MAX_SIZE_TEST = 1000
        cfg.INPUT.FORMAT = "BGR"
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        outputs = self.predictor(img)
        prediction = outputs["instances"].pred_masks.cpu().numpy()
        contours = []
        for pred in prediction:
            contour_list = get_contours_from_mask(pred)
            contours.append(get_larger_contour(contour_list))
        return contours


OOV_TOKEN = "<OOV>"
CTC_BLANK = "<BLANK>"


def get_char_map(alphabet):
    """Make from string alphabet character2int dict.
    Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [
                    self.char_map[char]
                    if char in self.char_map
                    else self.char_map[OOV_TOKEN]
                    for char in word
                ]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        dec_words = []
        for word in enc_word_list:
            word_chars = ""
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                        char_enc != self.char_map[OOV_TOKEN]
                        and char_enc != self.char_map[CTC_BLANK]
                        # idx > 0 to avoid selecting [-1] item
                        and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        return image


def get_val_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1)
    ])
    return transforms


def get_resnet34_backbone(pretrained=True):
    m = torchvision.models.resnet34(pretrained=pretrained)
    input_conv = nn.Conv2d(1, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(
            self,
            number_class_symbols,
            time_feature_count=256,
            lstm_hidden=256,
            lstm_len=2,
    ):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = get_val_transforms(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


class OcrPredictor:
    def __init__(self, model_path, config, device="cuda"):
        self.tokenizer = Tokenizer(config["alphabet"])
        self.device = torch.device(device)
        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config["image"]["height"],
            width=config["image"]["width"],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(
                f"Input must contain np.ndarray, "
                f"tuple or list, found {type(images)}."
            )

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


def get_image_visualization(img, pred_data, fontpath, font_koef=50):
    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h / font_koef))
    empty_img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data["predictions"]:
        polygon = prediction["polygon"]
        pred_text = prediction["text"]
        cv2.drawContours(img, np.array([polygon]), -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array([polygon]))
        draw.text((x, y), pred_text, fill=0, font=font)

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y: y + h, x: x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


class PiepleinePredictor:
    def __init__(self, segm_model_path, ocr_model_path, ocr_config):
        self.segm_predictor = SEGMpredictor(model_path=segm_model_path)
        self.ocr_predictor = OcrPredictor(model_path=ocr_model_path, config=ocr_config)

    def __call__(self, img):
        output = {"predictions": []}
        contours = self.segm_predictor(img)
        for contour in contours:
            if contour is not None:
                crop = crop_img_by_polygon(img, contour)
                pred_text = self.ocr_predictor(crop)
                output["predictions"].append(
                    {
                        "polygon": [[int(i[0][0]), int(i[0][1])] for i in contour],
                        "text": pred_text,
                    }
                )
        return output


def get_pipeline_predictor():
    return PiepleinePredictor(
        segm_model_path=SEGM_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        ocr_config=CONFIG_JSON,
    )


def get_predict(name):
    pipeline_predictor = get_pipeline_predictor()
    image = cv2.imread(name)
    result = pipeline_predictor(image)
    vis = get_image_visualization(image, result, 'times.ttf')
    img = Image.fromarray(vis, 'RGB')
    img.save(f'scans/{name}')


def main():
    pipeline_predictor = get_pipeline_predictor()
    pred_data = {}
    for img_name in tqdm(os.listdir(TEST_IMAGES_PATH)):
        image = cv2.imread(os.path.join(TEST_IMAGES_PATH, img_name))
        pred_data[img_name] = pipeline_predictor(image)

    with open(SAVE_PATH, "w") as f:
        json.dump(pred_data, f)
