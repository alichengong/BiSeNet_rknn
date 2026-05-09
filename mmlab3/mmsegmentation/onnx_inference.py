import argparse
import cv2
import numpy as np
import onnxruntime

def preprocess(image: np.ndarray, input_size: tuple = (640, 640)) -> tuple:
    """
    图像预处理：符合 BiSeNet V2 COCO 训练配置
    - Resize (保持宽高比 + padding 或直接 resize)
    - HWC → CHW
    - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    # BGR → RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize 到输入尺寸 (直接 resize，BiSeNet V2 通常使用固定输入)
    image_resized = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_resized.astype(np.float32) / 255.0 - mean) / std
    
    # HWC → CHW → add batch dimension
    image_tensor = np.transpose(image_normalized, (2, 0, 1))[np.newaxis, :, :, :]
    
    return image_tensor

def postprocess(pred: np.ndarray, original_size: tuple) -> np.ndarray:
    pred_map_squeezed = pred.squeeze()
    seg_map = cv2.resize(
      pred_map_squeezed.astype(np.uint8), 
      (original_size[1], original_size[0]), 
      interpolation=cv2.INTER_NEAREST)

    return seg_map

def get_fixed_palette(num_classes: int = 117) -> np.ndarray:
    np.random.seed(1234)
    palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    return palette

def visualize(image: np.ndarray, seg_map: np.ndarray, palette: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    可视化：原图 + 半透明分割掩码叠加
    """
    # 获取分割图的颜色映射
    seg_rgb = palette[seg_map]  # [H, W, 3]
    print('shapes: ', palette.shape, seg_map.shape, seg_rgb.shape)

    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    blended = cv2.addWeighted(image, 1 - alpha, seg_bgr, alpha, 0)
    return blended
 
def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # 1. 加载 ONNX 模型
    print(f"🔍 加载模型: {args.model}")
    sess = onnxruntime.InferenceSession(args.model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"   输入: {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}")
    print(f"   输出: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
    
    # 2. 读取并预处理图像
    print(f"🖼️  读取图像: {args.image_path}")
    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        raise ValueError(f"❌ 无法读取图片: {image_path}")
    original_size = image_bgr.shape[:2]  # (H, W)

    input_shape = tuple(map(int, args.input_shape.split(',')))
    input_tensor = preprocess(image_bgr, input_shape)
    print(f"   预处理后输入形状: {input_tensor.shape}")

    # 3. ONNX 推理
    print("🚀 执行推理...")
    predictions = sess.run([output_name], {input_name: input_tensor})[0]
    print(f"   输出形状: {predictions.shape}")

    # 4. 后处理
    print("🔄 后处理...")
    seg_map = postprocess(predictions, original_size)
    
    # 5. 可视化
    print("🎨 生成可视化结果...")
    palette = get_fixed_palette(19)
    overlay = visualize(image_bgr, seg_map, palette, 0.3)

    # 6. 保存结果
    cv2.imwrite(args.output_path, overlay)
    print(f"✅ 结果已保存: {args.output_path}")