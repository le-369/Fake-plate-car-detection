from ultralytics import YOLO
import cv2

car_model = YOLO(r'models\yolo_car_detector.pt')  # 汽车检测模型
belt_model = YOLO(r'models\yolo_car_belt_detector.pt')  # 安全带检测模型

# 输入单张图片
input_image_path = r"test.jpg"
output_image_path = 'test_result\\belt_result.png'  # 输出路径

# 读取输入图片
image = cv2.imread(input_image_path)
if image is None:
    raise ValueError(f"Cannot load image at {input_image_path}")

# 阶段 1: 汽车检测并裁剪 ROI
car_results = car_model.predict(source=image, conf=0.5)
if car_results[0].boxes:
    car_box = car_results[0].boxes.xyxy[0]  # 取第一个检测到的汽车框 [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, car_box[:4])
    roi = image[y1:y2, x1:x2]

    # 阶段 2: 安全带检测
    belt_results = belt_model.predict(source=roi, conf=0.3)  # 初始检测阈值较低
    if belt_results[0].boxes:
        # 遍历所有检测到的目标
        for box, cls, conf in zip(belt_results[0].boxes.xyxy, belt_results[0].boxes.cls, belt_results[0].boxes.conf):
            x1_roi, y1_roi, x2_roi, y2_roi = map(int, box)
            confidence = conf.item()

            # 确定标签
            threshold = 0.5  # 置信度阈值
            if confidence >= threshold:
                label = 'with_belt' if int(cls) == 1 else 'no_belt'
                color = (0, 255, 0) if label == 'with_belt' else (0, 0, 255)  # 绿色: 佩戴，红色: 未佩戴
            else:
                label = 'uncertain'
                color = (0, 165, 255)  # 橙色表示不确定

            # 绘制检测框和标签
            cv2.rectangle(roi, (x1_roi, y1_roi), (x2_roi, y2_roi), color, 2)
            cv2.putText(roi, f'{label} {confidence:.2f}', (x1_roi, y1_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 将标注后的 ROI 粘贴回原始图像
    image[y1:y2, x1:x2] = roi

    cv2.imwrite(output_image_path, image)
    print(f"Prediction saved to {output_image_path}")

else:
    print("No car detected in the image!")