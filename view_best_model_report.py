"""
View Best Model Report - RT-DETR Training Results
This script analyzes training results and displays comprehensive metrics for the best performing model.
"""
import pandas as pd
import os
from pathlib import Path


def find_latest_experiment():
    """Find the latest experiment directory"""
    runs_dir = Path('runs/detect/runs/train')
    if not runs_dir.exists():
        # Try alternative path
        runs_dir = Path('runs/train')

    if not runs_dir.exists():
        print("Error: No training runs found!")
        return None

    # Get all experiment directories
    experiments = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not experiments:
        print("Error: No experiment directories found!")
        return None

    # Sort by modification time and get the latest
    latest = sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return latest


def load_results(experiment_dir):
    """Load results.csv from experiment directory"""
    results_file = experiment_dir / 'results.csv'
    if not results_file.exists():
        print(f"Error: results.csv not found in {experiment_dir}")
        return None

    df = pd.read_csv(results_file)
    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_metrics_table(row, epoch):
    """Print metrics in a formatted table"""
    print(f"\nEpoch: {int(epoch)}")
    print("-" * 70)

    # Detection Metrics
    print("\nDetection Metrics:")
    print(f"  Precision (B):        {row['metrics/precision(B)']:.4f}")
    print(f"  Recall (B):           {row['metrics/recall(B)']:.4f}")
    print(f"  mAP50 (B):            {row['metrics/mAP50(B)']:.4f}")
    print(f"  mAP50-95 (B):         {row['metrics/mAP50-95(B)']:.4f}  ⭐ PRIMARY METRIC")

    # Training Losses
    print("\nTraining Losses:")
    print(f"  GIoU Loss:            {row['train/giou_loss']:.5f}")
    print(f"  Classification Loss:  {row['train/cls_loss']:.5f}")
    print(f"  L1 Loss:              {row['train/l1_loss']:.5f}")

    # Validation Losses
    print("\nValidation Losses:")
    print(f"  GIoU Loss:            {row['val/giou_loss']:.5f}")
    print(f"  Classification Loss:  {row['val/cls_loss']:.5f}")
    print(f"  L1 Loss:              {row['val/l1_loss']:.5f}")

    # Learning Rates
    print("\nLearning Rates:")
    print(f"  LR pg0:               {row['lr/pg0']:.6f}")
    print(f"  LR pg1:               {row['lr/pg1']:.6f}")
    print(f"  LR pg2:               {row['lr/pg2']:.6f}")


def display_report(experiment_dir=None):
    """Display comprehensive report of the best model"""

    # Find experiment directory
    if experiment_dir is None:
        experiment_dir = find_latest_experiment()
        if experiment_dir is None:
            return

    print_header("RT-DETR BEST MODEL REPORT")
    print(f"\nExperiment: {experiment_dir.name}")
    print(f"Location: {experiment_dir}")

    # Load results
    df = load_results(experiment_dir)
    if df is None:
        return

    # Find best model based on mAP50-95 (most important metric for object detection)
    best_idx = df['metrics/mAP50-95(B)'].idxmax()
    best_row = df.iloc[best_idx]
    best_epoch = int(best_row['epoch'])

    print_header(f"BEST MODEL (Epoch {best_epoch})")
    print_metrics_table(best_row, best_epoch)

    # Training Summary
    print_header("TRAINING SUMMARY")
    print(f"\nTotal Epochs:         {len(df)}")
    print(f"Best Epoch:           {best_epoch}")
    print(f"Best mAP50-95:        {best_row['metrics/mAP50-95(B)']:.4f}")
    print(f"Max Precision:        {df['metrics/precision(B)'].max():.4f} (Epoch {int(df.loc[df['metrics/precision(B)'].idxmax(), 'epoch'])})")
    print(f"Max Recall:           {df['metrics/recall(B)'].max():.4f} (Epoch {int(df.loc[df['metrics/recall(B)'].idxmax(), 'epoch'])})")
    print(f"Max mAP50:            {df['metrics/mAP50(B)'].max():.4f} (Epoch {int(df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch'])})")

    # Model Weights
    print_header("MODEL WEIGHTS")
    best_weights = experiment_dir / 'weights' / 'best.pt'
    last_weights = experiment_dir / 'weights' / 'last.pt'

    print(f"\nBest Model:  {best_weights}")
    print(f"             Exists: {best_weights.exists()}")
    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"             Size: {size_mb:.2f} MB")

    print(f"\nLast Model:  {last_weights}")
    print(f"             Exists: {last_weights.exists()}")
    if last_weights.exists():
        size_mb = last_weights.stat().st_size / (1024 * 1024)
        print(f"             Size: {size_mb:.2f} MB")

    # Available Visualizations
    print_header("AVAILABLE VISUALIZATIONS")
    viz_files = [
        'results.png',
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'BoxF1_curve.png',
        'BoxPR_curve.png',
        'BoxP_curve.png',
        'BoxR_curve.png',
    ]

    for viz in viz_files:
        viz_path = experiment_dir / viz
        if viz_path.exists():
            print(f"  ✓ {viz}")
        else:
            print(f"  ✗ {viz} (not found)")

    # Improvement over epochs
    print_header("TRAINING PROGRESSION")
    print("\nmAP50-95 by Epoch:")
    print("-" * 70)
    print(f"{'Epoch':>6} {'mAP50-95':>10} {'Change':>10} {'Status':>15}")
    print("-" * 70)

    for idx, row in df.iterrows():
        epoch = int(row['epoch'])
        map_val = row['metrics/mAP50-95(B)']

        if idx == 0:
            change = 0.0
            change_str = "-"
        else:
            change = map_val - df.iloc[idx-1]['metrics/mAP50-95(B)']
            change_str = f"{change:+.4f}"

        status = "⭐ BEST" if epoch == best_epoch else ""
        print(f"{epoch:>6} {map_val:>10.4f} {change_str:>10} {status:>15}")

    print("\n" + "="*70)
    print(f"Report generated successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    display_report()
