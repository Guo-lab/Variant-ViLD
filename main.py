import clip
import yaml
import numpy as np
from easydict import EasyDict
from scipy.special import softmax
import shutil
from pathlib import Path
import pdb

from postencoder.drawing import draw_output
from postencoder.text_encoder import process_label
from postencoder.visual_encoder import process_image


def main(CONFIG):

    clip.available_models()
    model, _ = clip.load("ViT-B/32")
    if not CONFIG.run_on_gpu:
        model.cpu()

    (category_embedding,
        category_names,
        category_indices) = process_label(model, CONFIG)
    
    ##TODO Investigate 
    (detection_roi_scores,
        detection_boxes,
        detection_masks,
        detection_visual_feat,
        rescaled_detection_boxes,
        valid_indices,
        image_info) = process_image(CONFIG)

    raw_scores = detection_visual_feat.dot(category_embedding.T)
    if CONFIG.use_softmax:
        scores_all = softmax(CONFIG.temperature * raw_scores, axis=-1)
    else:
        scores_all = raw_scores

    save_path = Path(CONFIG.save_path)
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir()

    draw_output(
        CONFIG,
        category_names = category_names,
        rescaled_detection_boxes = rescaled_detection_boxes,
        detection_masks = detection_masks,
        valid_indices = valid_indices,
        numbered_category_indices = category_indices,
        scores_all= scores_all,
        detection_roi_scores=detection_roi_scores,
        image_info=image_info)


if __name__ == "__main__":
    with open("config/demo.yaml", "r") as file:
        CONFIG = yaml.load(file, Loader=yaml.SafeLoader)
        CONFIG = EasyDict(CONFIG)
    main(CONFIG)