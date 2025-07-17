import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# 配置参数
YOLO_MODEL_PATH = r"models\yolo_car_detector.pt"
COLOR_MODEL_PATH = r"models\vehicle_color.hdf5"
TEST_IMAGE_PATH = r"test.jpg"
OUTPUT_IMAGE_PATH = r"test_result\color_result.png"
IMAGE_SIZE = (64, 64)
COLORS = ["black", "blue", "brown", "green", "red", "silver", "white", "yellow"]

def load_models():
    """加载 YOLO 检测模型和颜色分类模型"""
    yolo_model = YOLO(YOLO_MODEL_PATH)
    color_model = tf.keras.models.load_model(COLOR_MODEL_PATH, compile=False)
    return yolo_model, color_model

def preprocess_image(image):
    """预处理图像以用于颜色分类"""
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def predict_vehicle_color(image, color_model):
    """预测裁剪车辆区域的颜色"""
    processed = preprocess_image(image)
    preds = color_model.predict(processed, verbose=0)[0]
    color_idx = np.argmax(preds)
    predicted_color = COLORS[color_idx]
    confidence = preds[color_idx]
    return predicted_color, confidence

def main():
    # 加载模型
    yolo_model, color_model = load_models()
    
    # 读取输入图像
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print(f"无法读取图片: {TEST_IMAGE_PATH}")
        return
    
    # 第一阶段：YOLO 检测车辆
    results = yolo_model.predict(TEST_IMAGE_PATH, imgsz=640)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    if len(boxes) == 0:
        print("未检测到车辆")
        cv2.imwrite(OUTPUT_IMAGE_PATH, image)
        return
    
    # 选择置信度最高的边界框
    best_box_idx = np.argmax(confs)
    bbox = boxes[best_box_idx].astype(int)
    x1, y1, x2, y2 = bbox
    
    # 裁剪车辆区域
    cropped_vehicle = image[y1:y2, x1:x2]
    
    # 第二阶段：颜色预测
    predicted_color, confidence = predict_vehicle_color(cropped_vehicle, color_model)
    
    # 绘制结果
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{predicted_color} ({confidence:.1%})"
    cv2.putText(image, text, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)
    print(f"预测结果: {predicted_color} ({confidence:.1%})")
    print(f"结果已保存到: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()