"""
Run inference with trained RT-DETR model
"""
from ultralytics import RTDETR
import cv2
from pathlib import Path


def run_inference(model_path, image_source, output_dir='runs/detect'):
    """
    Run inference on images

    Args:
        model_path: Path to trained model weights
        image_source: Path to image, folder, or video
        output_dir: Output directory for results
    """
    # Load the model
    model = RTDETR(model_path)

    # Run inference
    results = model.predict(
        source=image_source,
        save=True,
        project=output_dir,
        name='predict',
        conf=0.25,  # Confidence threshold
        iou=0.45,   # IoU threshold for NMS
        imgsz=640,
        show_labels=True,
        show_conf=True,
    )

    # Print results
    for i, result in enumerate(results):
        print(f"\nImage {i + 1}:")
        print(f"  Boxes: {len(result.boxes)}")
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"    Class: {cls}, Confidence: {conf:.2f}")

    return results


def validate_model(model_path, data_yaml='data.yaml'):
    """Validate model on validation set"""
    model = RTDETR(model_path)
    metrics = model.val(data=data_yaml)

    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return metrics


if __name__ == "__main__":
    # Example: Run inference on test images
    run_inference(
        model_path='runs/train/rtdetr_experiment/weights/best.pt',
        image_source='dataset_yolo/images/val'
    )

    # Or validate on the validation set
    # validate_model('runs/train/rtdetr_experiment/weights/best.pt')
