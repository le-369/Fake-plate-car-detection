import torch
import cv2
import numpy as np
from ultralytics.nn.tasks import attempt_load_weights
from plate_recognition.plate_rec import get_plate_result, init_model
from plate_recognition.double_plate_split_merge import get_split_merge
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        r"inference\license_plate\plate_recognition\platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def four_point_transform(image, pts):  # Perspective transform for plate region
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def letter_box(img, size=(640, 640)):  # YOLO preprocessing: letterbox
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    new_img = cv2.resize(img, (new_w, new_h))
    left = (size[1] - new_w) // 2
    top = (size[0] - new_h) // 2
    img = cv2.copyMakeBorder(new_img, top, size[0] - top - new_h, left, size[1] - left - new_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top

def load_model(weights, device):  # Load YOLOv8 model
    return attempt_load_weights(weights, device=device)

def xywh2xyxy(det):  # Convert xywh to xyxy
    y = det.clone()
    y[:, 0] = det[:, 0] - det[:, 2] / 2
    y[:, 1] = det[:, 1] - det[:, 3] / 2
    y[:, 2] = det[:, 0] + det[:, 2] / 2
    y[:, 3] = det[:, 1] + det[:, 3] / 2
    return y

def my_nms(dets, iou_thresh):  # Non-Maximum Suppression
    y = dets.clone()
    index = torch.argsort(y[:, 4], descending=True)
    keep = []
    while index.size(0) > 0:
        i = index[0].item()
        keep.append(i)
        x1 = torch.maximum(y[i, 0], y[index[1:], 0])
        y1 = torch.maximum(y[i, 1], y[index[1:], 1])
        x2 = torch.minimum(y[i, 2], y[index[1:], 2])
        y2 = torch.minimum(y[i, 3], y[index[1:], 3])
        w = torch.maximum(torch.tensor(0).to(device), x2 - x1)
        h = torch.maximum(torch.tensor(0).to(device), y2 - y1)
        inter_area = w * h
        union_area1 = (y[i, 2] - y[i, 0]) * (y[i, 3] - y[i, 1])
        union_area2 = (y[index[1:], 2] - y[index[1:], 0]) * (y[index[1:], 3] - y[index[1:], 1])
        iou = inter_area / (union_area1 + union_area2 - inter_area)
        idx = torch.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(dets, r, left, top):  # Restore boxes to original image scale
    dets[:, [0, 2]] = (dets[:, [0, 2]] - left) / r
    dets[:, [1, 3]] = (dets[:, [1, 3]] - top) / r
    return dets

def post_processing(prediction, conf, iou_thresh, r, left, top):  # Post-processing
    prediction = prediction.permute(0, 2, 1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf
    x = prediction[xc]
    if not len(x):
        return []
    boxes = xywh2xyxy(x[:, :4])
    score, index = torch.max(x[:, 4:6], dim=-1, keepdim=True)
    x = torch.cat((boxes, score, x[:, 6:14], index), dim=1)
    keep = my_nms(x, iou_thresh)
    x = restore_box(x[keep], r, left, top)
    return x

def pre_processing(img, device, img_size=640):  # Pre-processing
    img, r, left, top = letter_box(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(img).to(device).float() / 255.0
    return img.unsqueeze(0), r, left, top

def det_rec_plate(img, img_ori, detect_model, plate_rec_model):  # Detect and recognize plate
    result_list = []
    img, r, left, top = pre_processing(img, device)
    predict = detect_model(img)[0]
    outputs = post_processing(predict, 0.3, 0.5, r, left, top)
    for output in outputs:
        result_dict = {}
        output = output.cpu().numpy()
        rect = output[:4].astype(int)
        label = int(output[-1])
        roi_img = img_ori[rect[1]:rect[3], rect[0]:rect[2]]
        if label:  # Double-layer plate
            roi_img = get_split_merge(roi_img)
        plate_number, _, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model, is_color=True)
        result_dict['plate_no'] = plate_number
        result_dict['plate_color'] = plate_color
        result_dict['rect'] = rect.tolist()
        result_dict['detect_conf'] = output[4]
        result_dict['color_conf'] = color_conf
        result_dict['plate_type'] = label
        result_list.append(result_dict)
    return result_list

def draw_result(orgimg, dict_list):  # Draw beautified results on image
    result_str = ""
    for result in dict_list:
        rect = result['rect']
        x, y, w, h = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        # Expand bounding box slightly for aesthetics
        rect = [max(0, int(x - 0.05 * w)), max(0, int(y - 0.05 * h)),
                min(orgimg.shape[1], int(rect[2] + 0.05 * w)), min(orgimg.shape[0], int(rect[3] + 0.05 * h))]
        
        # Prepare text
        result_p = f"{result['plate_no']} {result['plate_color']}{' 双层' if result['plate_type'] else ''}"
        result_str += result_p + " "
        
        # Draw rounded rectangle (approximated with thicker, softer lines)
        cv2.rectangle(orgimg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 200, 0), 2, lineType=cv2.LINE_AA)
        
        # Semi-transparent text background
        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = min(rect[0], orgimg.shape[1] - labelSize[0])
        text_y = int(rect[1] - 1.8 * labelSize[1])
        text_bg = (text_x, text_y, text_x + int(1.2 * labelSize[0]), text_y + int(1.5 * labelSize[1]))
        
        overlay = orgimg.copy()
        cv2.rectangle(overlay, (text_bg[0], text_bg[1]), (text_bg[2], text_bg[3]), (255, 255, 255), -1)
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, orgimg, 1 - alpha, 0, orgimg)
        
        # Draw text with a modern font style
        orgimg = cv2ImgAddText(orgimg, result_p, text_x, text_y, (0, 0, 0), 24)
    
    print(result_str)
    return orgimg

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detect_model = load_model(r'models\yolov8s.pt', device)
plate_rec_model = init_model(device, r'models\plate_rec_color.pth', is_color=True)
detect_model.eval()

img = cv2.imread(r"test.jpg")
result_list = det_rec_plate(img, img.copy(), detect_model, plate_rec_model)
result_img = draw_result(img, result_list)
cv2.imwrite('test_result\\plate_result.png', result_img)