from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")
from detectron2.projects import point_rend
import glob
import os
import torch

@torch.no_grad()
def automatic_masking(images_folder, result_folder, bounding_box=None):
    #COCO_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    POINTREND_MODEL = "../../../Downloads/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    POINTREND_WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(POINTREND_MODEL)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = POINTREND_WEIGHTS
    predictor = DefaultPredictor(cfg)

    image_files = [f for f in sorted(glob.glob(os.path.join(images_folder, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jepg')]
    for file in image_files:
        im = np.array(Image.open(file), dtype=np.uint8)
        print("Processing image {}".format(file))
        outputs = predictor(im)

        im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], 1), dtype=np.uint8)), axis=-1, dtype=np.uint8)
        for mask in outputs["instances"].pred_masks:
            im[:, :, -1] = np.clip(np.add(im[:, :, -1], np.array(mask.cpu(), dtype=np.uint8)), 0, 1) * 255

        roi_alpha = 0
        if bounding_box is None:
            roi_alpha = np.copy(im[:, :, -1])
        else:
            roi_alpha = np.copy(im[bounding_box[0, 1]:bounding_box[1, 1], bounding_box[0, 0]:bounding_box[1, 0], -1])
        if np.sum(roi_alpha) > 0:
            if bounding_box is not None:
                im[:, :, -1] = 0
                assert np.sum(roi_alpha) > 0
                im[bounding_box[0, 1]:bounding_box[1, 1], bounding_box[0, 0]:bounding_box[1, 0], -1] = roi_alpha
            out_file = file.split("/")[-1]
            Image.fromarray(im.astype(np.uint8)).save( os.path.join(result_folder, out_file))
