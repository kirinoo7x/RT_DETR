"""
Train RT-DETR model on custom dataset
"""
from ultralytics import RTDETR
import torch


def train_model():
    # Auto-detect best available device
    # Note: MPS (Apple Silicon) has incomplete support for RT-DETR operations
    if torch.cuda.is_available():
        device = 0  # CUDA GPU
    else:
        device = 'cpu'  # CPU (MPS not fully supported for RT-DETR)

    print(f"Using device: {device}")

    # Load a pretrained RT-DETR model
    model = RTDETR('rtdetr-l.pt')  # or rtdetr-x.pt for larger model

    # Train the model
    # Note: CPU training is slow. Consider reducing epochs or batch size for faster testing.
    results = model.train(
        data='data.yaml',
        epochs=50,  # Reduced for CPU training (was 100)
        imgsz=640,
        batch=4,  # Reduced batch size for CPU (was 8)
        device=device,
        project='runs/train',
        name='rtdetr_experiment',
        patience=25,  # Early stopping patience
        save=True,
        plots=True,
        workers=4,  # Number of dataloader workers
    )

    return results


if __name__ == "__main__":
    train_model()
