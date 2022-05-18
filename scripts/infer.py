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

from IPython.display import display
from PIL import Image

def cv2_imshow(a, convert_bgr_to_rgb=True):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))

# %%
config_path = '/home/zp-yang/git/nanodet/config/nanodet-plus-m-1.5x_416-cf.yml'
model_path = '/home/zp-yang/git/nanodet/trained/nanodet-plus-m-1.5x_416-cf/model_best/nanodet_model_best.pth'
load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)

predictor = Predictor(cfg, model_path, logger, device=device)
# %%

# image_path = '/home/zp-yang/git/nanodet/coco/val/JPEGImages/dataset-0-frame-630.jpg'
image_path = "/home/zp-yang/git/nanodet/test_data/test_img1.png"
image_path = "/home/zp-yang/git/nanodet/test_data/test_img2.png"


# meta, res = predictor.inference(image_path)

# result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35)
# imshow_scale = 1.0
# cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))
# %%
cap = cv2.VideoCapture("/home/zp-yang/git/nanodet/test_data/20220424_161731.mp4")
# cap = cv2.VideoCapture("/home/zp-yang/git/nanodet/test_data/20220424_163422.mp4")
if (cap.isOpened()):
    print("Error opening video stream or file")

frame_width = 2288
frame_height = 1080
fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
out = cv2.VideoWriter('test_data/output.mp4', fourcc, 20, (frame_width,frame_height))

while (cap.isOpened()):
    ret, frame = cap.read()
    # print(frame.shape)
    if ret == True:
        # result = frame
        meta, res = predictor.inference(frame)
        result = overlay_bbox_cv(meta["raw_img"][0], res[0], cfg.class_names, score_thresh=0.35)
        out.write(result)

        cv2.imshow("Detection Result", result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print("not getting any frame...")
        break
cap.release()
out.release()
cv2.destroyAllWindows()
# %%
