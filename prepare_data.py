"""
Convert segmentation masks to YOLO format bounding boxes for RT-DETR
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def mask_to_bbox(mask_path):
    """Convert a segmentation mask to bounding boxes in YOLO format"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    height, width = mask.shape

    # Find all unique non-zero values (different objects)
    unique_vals = np.unique(mask)
    unique_vals = unique_vals[unique_vals > 0]  # Remove background

    bboxes = []
    for val in unique_vals:
        # Create binary mask for this object
        binary_mask = (mask == val).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 10:  # Skip very small contours
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height

            # Class 0 (adjust if you have multiple classes)
            class_id = 0
            bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    return bboxes


def prepare_dataset(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, output_dir):
    """Prepare dataset in YOLO format"""
    output_dir = Path(output_dir)

    # Create directory structure
    train_img_out = output_dir / "images" / "train"
    val_img_out = output_dir / "images" / "val"
    train_label_out = output_dir / "labels" / "train"
    val_label_out = output_dir / "labels" / "val"

    for dir_path in [train_img_out, val_img_out, train_label_out, val_label_out]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process training data
    print("Processing training data...")
    train_images = sorted(os.listdir(train_img_dir))
    for img_name in tqdm(train_images):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(train_img_dir, img_name)
        mask_path = os.path.join(train_mask_dir, img_name)

        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {img_name}")
            continue

        # Copy image
        import shutil
        shutil.copy(img_path, train_img_out / img_name)

        # Convert mask to bboxes
        bboxes = mask_to_bbox(mask_path)

        # Save label file
        label_name = Path(img_name).stem + ".txt"
        with open(train_label_out / label_name, 'w') as f:
            f.write('\n'.join(bboxes))

    # Process test/validation data
    print("Processing test data...")
    test_images = sorted(os.listdir(test_img_dir))
    for img_name in tqdm(test_images):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(test_img_dir, img_name)
        mask_path = os.path.join(test_mask_dir, img_name)

        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {img_name}")
            continue

        # Copy image
        import shutil
        shutil.copy(img_path, val_img_out / img_name)

        # Convert mask to bboxes
        bboxes = mask_to_bbox(mask_path)

        # Save label file
        label_name = Path(img_name).stem + ".txt"
        with open(val_label_out / label_name, 'w') as f:
            f.write('\n'.join(bboxes))

    print(f"\nDataset prepared in {output_dir}")
    print(f"Train images: {len(list(train_img_out.glob('*')))}")
    print(f"Val images: {len(list(val_img_out.glob('*')))}")


if __name__ == "__main__":
    prepare_dataset(
        train_img_dir="cc/Train/images",
        train_mask_dir="cc/Train/masks",
        test_img_dir="cc/Test/images",
        test_mask_dir="cc/Test/masks",
        output_dir="dataset_yolo"
    )
