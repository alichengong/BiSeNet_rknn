# export_yolox_onnx.py
from mmengine.config import Config
from mmdeploy.apis import torch2onnx
from mmdeploy.core import FUNCTION_REWRITER

import torch

deploy_cfg = Config.fromfile('configs/mmseg/segmentation_onnxruntime_static-512x512.py')
deploy_cfg.onnx_config.input_shape = [1024, 1024]

# deploy_cfg.codebase_config.model_type = 'raw'
# deploy_cfg.codebase_config.pop('post_processing', None)

model_cfg = Config.fromfile('../mmsegmentation/configs/bisenetv2/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py')
model_ckpt = 'bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth'

img_path = 'demo/resources/cityscapes.png'
work_dir = './onnx'

torch2onnx(
  img=img_path,
  model_cfg=model_cfg,
  deploy_cfg=deploy_cfg,
  device='cpu',
  work_dir=work_dir,
  model_checkpoint=model_ckpt,
  save_file='bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.onnx')