import argparse
import cv2
import numpy as np
import onnxruntime
from typing import List, Optional

COCO_STUFF_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner",
    "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard",
    "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter",
    "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble",
    "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other",
    "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves",
    "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net",
    "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield",
    "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea",
    "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw",
    "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable",
    "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile",
    "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"
]

assert len(COCO_STUFF_CLASSES) == 171, f"Expected 171 classes, got {len(COCO_STUFF_CLASSES)}"

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

def postprocess(pred: np.ndarray, original_size: tuple, confidence_threshold: float = 0.5) -> np.ndarray:
    exp_logits = np.exp(pred)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    max_probs = np.max(probs, axis=1)
    pred_map = np.argmax(probs, axis=1)

    max_probs_squeezed = np.squeeze(max_probs)
    pred_map_squeezed = np.squeeze(pred_map)

    low_conf_mask = max_probs_squeezed < confidence_threshold
    pred_map_squeezed[low_conf_mask] = 0

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

def display_coco_stuff_labels(
    image: np.ndarray,
    seg_map: np.ndarray,
    class_names: List[str],
    palette: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    font_scale: float = 0.45,
    min_area_ratio: float = 0.0005
) -> np.ndarray:
    """
    在分割区域上显示 COCO-Stuff 类别标签
    :param image: 原始图像 (H, W, 3)，支持 RGB 或 BGR
    :param seg_map: 分割图 (H, W)，值为 0-170 的类别索引
    :param class_names: 类别名称列表 (长度 171)
    :param palette: 颜色表 (N, 3)，默认自动生成
    :param alpha: 掩码透明度 (0~1)
    :param font_scale: 字体大小
    :param min_area_ratio: 最小区域面积占全图比例，过滤噪点碎片
    """
    if palette is None:
        palette = get_fixed_palette(len(class_names))

    img_bgr = image.copy()
    h, w = img_bgr.shape[:2]
    min_area = int(h * w * min_area_ratio)

    # 1️⃣ 生成半透明彩色掩码
    color_mask = np.zeros_like(img_bgr)
    unique_classes = np.unique(seg_map)
    for cls_id in unique_classes:
        if cls_id == 0 or cls_id >= len(class_names):
            continue
        color_mask[seg_map == cls_id] = palette[cls_id]

    blended = cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)

    # 2️⃣ 在连通区域质心绘制标签
    result = blended.copy()

    for cls_id in unique_classes:
        if cls_id == 0 or cls_id >= len(class_names):
            continue
            
        mask = (seg_map == cls_id).astype(np.uint8)
        if np.sum(mask) < min_area:
          continue

        # 找轮廓 & 取最大连通域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
          continue

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
          continue

        # 计算质心
        M = cv2.moments(largest)
        if M["m00"] == 0:
          continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        # 自动对比色
        bg_color = tuple(int(x) for x in palette[cls_id])
        brightness = 0.299*bg_color[0] + 0.587*bg_color[1] + 0.114*bg_color[2]
        txt_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        label = class_names[cls_id]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        
        # 防越界
        x = max(2, min(cx - tw // 2, w - tw - 2))
        y = max(th + 2, min(cy + th // 2, h - 2))
        
        # 绘制背景框 + 文字
        cv2.rectangle(result, (x, y - th - 3), (x + tw + 3, y + 1), bg_color, -1)
        cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, 1, cv2.LINE_AA)

    return result

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
    seg_map = postprocess(predictions, original_size, confidence_threshold=0.3)
    
    # 5. 可视化
    print("🎨 生成可视化结果...")
    palette = get_fixed_palette(171)
    overlay = visualize(image_bgr, seg_map, palette, 0.6)
    result_combined = display_coco_stuff_labels(
      image=image_bgr,
      seg_map=seg_map,
      class_names=COCO_STUFF_CLASSES,
      alpha=0.35,          # 掩码透明度
      font_scale=0.4,      # 字体大小
      min_area_ratio=0.002 # 过滤 <0.1% 面积的碎片区域
    )

    # 6. 保存结果
    cv2.imwrite(args.output_path, result_combined)
    print(f"✅ 结果已保存: {args.output_path}")