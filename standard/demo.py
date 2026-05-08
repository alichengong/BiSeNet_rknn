
import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

torch.set_grad_enabled(False)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)

np.random.seed(1234)
palette = np.random.randint(0, 256, (171, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.to(device)

# prepare data
to_tensor = T.ToTensor(
    mean=(0.485, 0.456, 0.406), # coco, rgb
    std=(0.229, 0.224, 0.225)
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).to(device)

# shape divisor
org_size = im.size()[2:]
new_size = [640, 640]

# inference
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
out = net(im)[0]
out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
print('pytorch intputs: ', im.shape, torch.min(im), torch.max(im))
print('pytorch outputs: ', out.shape, torch.min(out), torch.max(out))
out = out.argmax(dim=1)

# visualize
out = out.squeeze().detach().cpu().numpy()
print('pytorch seg_map: ', out.shape, np.min(out), np.max(out))
pred = palette[out]

seg_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
cv2.imwrite('./res.jpg', seg_bgr)
