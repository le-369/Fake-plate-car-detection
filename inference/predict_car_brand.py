import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
from ultralytics import YOLO
import torch.nn as nn
from torchvision import models

def load_models(yolo_model_path, brand_model_path, label_map_path, device):
    yolo_model = YOLO(yolo_model_path)
    with open(label_map_path, 'r', encoding='utf-8') as f:
        brand_to_idx = json.load(f)
    idx_to_brand = {v: k for k, v in brand_to_idx.items()}
    num_classes = len(brand_to_idx)
    brand_model = get_efficientnet_b3(num_classes=num_classes, pretrained=False)
    brand_model.load_state_dict(torch.load(brand_model_path, map_location=device))
    brand_model = brand_model.to(device).eval()
    return yolo_model, brand_model, idx_to_brand

def get_efficientnet_b3(num_classes, pretrained=True):
    # 加载预训练的 EfficientNet-B3 模型
    model = models.efficientnet_b3(pretrained=pretrained)
    
    # 修改分类层以适配车品牌数量
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),  # 添加 dropout 防止过拟合
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    return model

def predict_two_stage(image_path, yolo_model, brand_model, idx_to_brand, device):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.resize((224, 224), Image.BILINEAR)),  # 简化的 Resize，无需填充
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    results = yolo_model.predict(image_path, imgsz=640, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    if len(boxes) == 0:
        return None, None, image, None
    
    best_box_idx = np.argmax(confs)
    bbox = boxes[best_box_idx].astype(int)
    car_region = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    car_tensor = transform(car_region).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = brand_model(car_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        pred_prob = probabilities[pred_idx].item()
        pred_brand = idx_to_brand[pred_idx]
    
    return pred_brand, pred_prob, image, bbox

def draw_prediction(image, brand, prob, bbox):
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        draw.rectangle(bbox.tolist(), outline=(0, 255, 0), width=3)
    
    try:
        font = ImageFont.truetype("simsun.ttc", size=40)
    except:
        font = ImageFont.load_default()
    
    text = f"{brand} ({prob*100:.2f}%)" if brand else "无车辆"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x, text_y, box_margin = 10, 10, 5
    
    draw.rectangle(
        (text_x - box_margin, text_y - box_margin, 
         text_x + text_width + box_margin, text_y + text_height + box_margin),
        fill=(0, 0, 0)
    )
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    return image

# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model, brand_model, idx_to_brand = load_models(
    r"models\yolo_car_detector.pt",
    r"models\car_brand_classifier.pth",
    r"models\label_map.json",
    device
)
pred_brand, pred_prob, image, bbox = predict_two_stage(
    r"test.jpg",
    yolo_model, brand_model, idx_to_brand, device
)
annotated_image = draw_prediction(image, pred_brand, pred_prob, bbox)
annotated_image.save('test_result\\brand_result.png')
if pred_brand:
    print(f"预测结果: {pred_brand}，概率: {pred_prob*100:.2f}%")
else:
    print("未检测到车辆")