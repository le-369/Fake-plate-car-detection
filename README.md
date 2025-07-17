# 🚗 Fake Plate Car Detection System

A robust system for detecting fake plate vehicles and monitoring driver safety (e.g., seat belt usage), designed for deployment in traffic control scenarios.

## 📌 Project Introduction

Fake plate cars—vehicles using duplicated license plates—are commonly detected via bayonet cameras in intelligent traffic systems. These vehicles often share identical plate numbers while differing in key visual characteristics such as **color**, **brand**, **model**, and **body shape**.

This project presents a two-stage feature detection framework that extracts and analyzes multiple attributes of vehicles from captured images to detect inconsistencies indicative of plate cloning. Additionally, a **seat belt inference module** is integrated to monitor whether drivers are complying with road safety requirements.

## 📦 Prerequisites & Installation

### 🔧 Prerequisites

- **Python**: Version 3.8 or higher
- **Git**: For cloning the repository
- **Dependencies**: Listed in `requirements.txt`

### 📥 Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/le-369/Fake-plate-car-detection.git
   cd Fake-plate-car-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained model weights:

- Access the Google Drive folder: [Model Weights](https://drive.google.com/drive/folders/10aQi223yn6hZEjfldmuow7iQrSsMu_uW?usp=drive_link)
- Download all weight files and place them in the `./weights` folder in the project directory
- Ensure the folder structure matches the expected paths in the inference scripts (e.g.,` ./weights/car_belt.pth`, `./weights/car_brand.pth`, etc.).

## 📂 Project Structure

```bash
├── inference/             # Inference code
├── weights/               # Pre-trained and custom models
├── test_result/           # Reasoning result
├── requirements.txt       # Required packages
└── README.md              # Project documentation
```

## 🖼️ System Framework

The following diagram illustrates the architecture of the Fake Plate Car Detection System:

![图1](images\\net.bmp "Algorithm framework")

This framework outlines the two-stage detection process, including license plate recognition, vehicle attribute analysis, and seat belt inference.

We only did the inference on the CPU and generally achieved the following performance (depending on the car), but on average it was less than 300ms.

![图2](images\\net1.bmp "Execution details")


## 🚀 How to Run

Run the detection system with the following command:

```bash
# Detect seat belt usage
python inference//predict_car_belt.py
# Detect vehicle brand
python inference//predict_car_brand.py
# Detect vehicle color
python inference//predict_car_color.py
# Detect vehicle type
python inference//predict_car_type.py
# Detect and recognize license plate
python inference//license_plate//predict_rec_plate.py
```



## 📈 Features

- ✅ License plate detection and OCR
- ✅ Vehicle model and brand recognition
- ✅ Color classification
- ✅ Seat belt detection
- ✅ Modular and extensible architecture



## 📚 References
https://github.com/we0091234/yolov8-plate

https://github.com/ultralytics/ultralytics



## 📮 Contact

For questions, feedback, or collaboration, feel free to reach out:

**chime**  
📧 [HELLOLE_369@126.com]

