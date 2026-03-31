# RT-DETR Object Detection

This project uses RT-DETR (Real-Time Detection Transformer) for object detection on your custom dataset.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Dataset
Convert your segmentation masks to YOLO format bounding boxes:
```bash
python prepare_data.py
```

This will create a `dataset_yolo` folder with the proper structure.

### Step 2: Train Model
Train the RT-DETR model:
```bash
python train_rtdetr.py
```

Training results will be saved in `runs/train/rtdetr_experiment/`

### Step 3: Run Inference
Run inference on test images:
```bash
python inference.py
```

Results will be saved in `runs/detect/predict/`

## Configuration

- **Model size**: Edit `train_rtdetr.py` to use `rtdetr-l.pt` (large) or `rtdetr-x.pt` (extra large)
- **Training params**: Modify epochs, batch size, image size in `train_rtdetr.py`
- **Classes**: Update `data.yaml` with your actual class names

## Dataset Structure

```
dataset_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Notes

- RT-DETR is a real-time transformer-based object detector
- The model automatically converts segmentation masks to bounding boxes
- Adjust confidence threshold in `inference.py` for detection sensitivity
