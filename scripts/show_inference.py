#%%
import os
import cv2
import torch
from demo.demo import Predictor
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda')
# %%
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# %%
from nanodet.util import cfg, load_config, Logger

from nanodet.util import overlay_bbox_cv

# %%
config_path = '/home/visnet/git/nanodet/config/nanodet-plus-m-1.5x_416-cf.yml'
model_path = '/home/visnet/git/nanodet/trained/nanodet_model_best.pth'
load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)

predictor = Predictor(cfg, model_path, logger, device=device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # result = frame
        meta, res = predictor.inference(frame)
        result = overlay_bbox_cv(meta["raw_img"][0], res[0], cfg.class_names, score_thresh=0.35)

        cv2.imshow("Detection Result", result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print("not getting any frame...")
        break
cap.release()
cv2.destroyAllWindows()