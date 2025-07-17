"""
车辆检测与分类集成系统
功能：先检测图片中的车辆位置，再识别车辆类型
"""
import cv2
import numpy as np
import os
import tempfile
import shutil
from ultralytics import YOLO
from keras_preprocessing.image import img_to_array
import tensorflow as tf

# 车辆检测模型路径
YOLO_MODEL_PATH = r"models\yolo_car_detector.pt"
# 车辆类型分类模型路径
TYPE_MODEL_PATH = r"models\vehicle_type.hdf5"
# 测试图片路径
TEST_IMAGE_PATH = r"test.jpg"
# 输出图片路径
OUTPUT_IMAGE_PATH = "test_result\\type_result.png"
# 车辆类型列表
CLASSES = ["bus", "car", "minibus", "truck"]
# 车辆图像尺寸（用于分类模型）
VEHICLE_WIDTH = 32
VEHICLE_HEIGHT = 32

# ----------------------
# 模型加载
# ----------------------
def load_models():
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # 创建临时目录并加载类型分类模型
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, 'vehicle_type.hdf5')
    shutil.copy2(TYPE_MODEL_PATH, temp_model_path)
    
    try:
        type_model = tf.keras.models.load_model(temp_model_path)
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型出错: {e}")
        try:
            type_model = tf.keras.models.load_model(temp_model_path, compile=False)
            print("备用方法加载成功")
        except Exception as inner_e:
            print(f"备用方法失败: {inner_e}")
            raise
    
    return yolo_model, type_model, temp_dir

# ----------------------
# 车辆类型预测
# ----------------------
def predict_vehicle_type(cropped_image, type_model):
    """预测裁剪出的车辆图像类型"""
    # 转换为灰度图
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # 调整大小
    roi = cv2.resize(gray, (VEHICLE_WIDTH, VEHICLE_HEIGHT))
    # 归一化
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # 预测
    predictions = type_model.predict(roi)[0]
    result = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
    label = max(result, key=lambda x: result[x])
    
    return label, result

def main():

    yolo_model, type_model, temp_dir = load_models()
    
    try:
        image = cv2.imread(TEST_IMAGE_PATH)
        if image is None:
            raise ValueError(f"无法读取图片: {TEST_IMAGE_PATH}")
        
        # 使用YOLO检测车辆
        results = yolo_model.predict(TEST_IMAGE_PATH, imgsz=640)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        if len(boxes) == 0:
            print("未检测到车辆")
            return
        
        # 选择置信度最高的检测框
        best_box_idx = np.argmax(confs)
        bbox = boxes[best_box_idx].astype(int)
        
        # 裁剪车辆区域
        x1, y1, x2, y2 = bbox
        cropped_vehicle = image[y1:y2, x1:x2]
        
        # 预测车辆类型
        vehicle_type, probabilities = predict_vehicle_type(cropped_vehicle, type_model)
        
        # 在原始图像上绘制结果
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, vehicle_type, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 保存结果
        cv2.imwrite(OUTPUT_IMAGE_PATH, image)
        
        # 打印结果
        print("\n检测结果:")
        print(f"车辆位置: {bbox.tolist()}")
        print(f"车辆类型: {vehicle_type}")
        print("各类别概率:")
        for cls, prob in probabilities.items():
            print(f"  {cls}: {prob:.4f}")
        print(f"\n结果已保存到: {OUTPUT_IMAGE_PATH}")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()