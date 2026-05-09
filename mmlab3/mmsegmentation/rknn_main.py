import cv2
import numpy as np
from rknn.api import RKNN
import torch
from torchvision.ops import nms

class rknn_api():
  def __init__(self, model_path):
    self.rknn = RKNN(verbose=True)
    self.model_path = model_path
    self.init_flag = False

  def build(self):
    self.rknn.config(
      mean_values=[[123.675, 116.28, 103.53]],
      std_values=[[58.395, 57.12, 57.375]],
      target_platform="rk3588")

    print('--> Loading model:', self.model_path)
    ret = self.rknn.load_onnx(
      model=self.model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    do_quant = True
    DATASET_PATH = '../datasets/calibration/img.txt'
    ret = self.rknn.build(do_quantization=do_quant, dataset=DATASET_PATH, auto_hybrid=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

  def export(self, output_path):
    print('--> Export rknn model')
    ret = self.rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('✅ Export rknn model done')

  def preproc(self, image: np.ndarray, input_size: tuple = (640, 640)) -> tuple:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_LINEAR)

    image_tensor = np.transpose(image_resized, (2, 0, 1))[np.newaxis, :, :, :]
    image_tensor = np.ascontiguousarray(image_tensor, dtype=np.float32)

    return image_tensor

  def postproc(self, pred: np.ndarray, original_size: tuple) -> np.ndarray:
    pred_map_squeezed = pred.squeeze()
    seg_map = cv2.resize(
      pred_map_squeezed.astype(np.uint8), 
      (original_size[1], original_size[0]), 
      interpolation=cv2.INTER_NEAREST)

    return seg_map

  def get_fixed_palette(self, num_classes: int = 117) -> np.ndarray:
    np.random.seed(1234)
    palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    return palette

  def visualize(self, image: np.ndarray, seg_map: np.ndarray, palette: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    seg_rgb = palette[seg_map]  # [H, W, 3]
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    blended = cv2.addWeighted(image, 1 - alpha, seg_bgr, alpha, 0)
    return blended

  def infer(self, img_path):
    if not self.init_flag:
      ret = self.rknn.init_runtime(target=None)
      if ret != 0:
        raise RuntimeError(f'❌ Init runtime failed (ret={ret})')
      self.init_flag = True

    img = cv2.imread(img_path)
    original_size = img.shape[:2]  # (H, W)
    
    image_tensor = self.preproc(img, (1024,1024))
    outputs = self.rknn.inference(inputs=[image_tensor], data_format='nchw')
    seg_map = self.postproc(outputs[0], original_size)

    palette = self.get_fixed_palette(19)
    overlay = self.visualize(img, seg_map, palette, 0.4)
    cv2.imwrite('rknn_seg.jpg', overlay)

  def __del__(self):
    self.rknn.release()

if __name__ == '__main__':
  rknnApi = rknn_api(model_path='bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.onnx')
  rknnApi.build()
  rknnApi.export(output_path='model_final_v2_coco.rknn')
  rknnApi.infer(img_path='cityscapes.png')