import numpy as np
from pathlib import Path
from PIL import Image
import tifffile as tiff
from tqdm import tqdm

def compute_threshold_from_dir(tiff_files):
    """Compute threshold as mean + 3 * std across all tiff images."""
    scores = []
    for tiff_path in tiff_files:
        score = tiff.imread(tiff_path).astype(np.float32)
        scores.append(score.flatten())
    all_scores = np.concatenate(scores)
    threshold = np.mean(all_scores) + 3 * np.std(all_scores)
    return threshold

def threshold_anomaly_map(anomaly_map, threshold):
    """Binarize the anomaly map using the given threshold."""
    return (anomaly_map > threshold).astype(np.uint8) * 255

def convert_all(input_root: str, output_root: str):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for category_dir in sorted(input_root.iterdir()):
        if not category_dir.is_dir():
            continue

        for subfolder in ['test_private', 'test_private_mixed']:
            input_subdir = category_dir / subfolder
            if not input_subdir.exists():
                continue

            tiff_files = sorted(input_subdir.glob("*.tiff"))
            if not tiff_files:
                continue

            threshold = compute_threshold_from_dir(tiff_files)
            print(f"[{category_dir.name}/{subfolder}] threshold = {threshold:.4f}")

            for tiff_path in tqdm(tiff_files, desc=f"{category_dir.name}/{subfolder}"):
                anomaly_map = tiff.imread(tiff_path).astype(np.float32)
                binary_mask = threshold_anomaly_map(anomaly_map, threshold)

                # 對應輸出路徑
                relative_path = tiff_path.relative_to(input_root).with_suffix(".png")
                output_path = output_root / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(binary_mask).save(output_path)

if __name__ == "__main__":
    input_root = "mvtec_results/anomaly_images"
    output_root = "mvtec_results/anomaly_images_thresholded"
    convert_all(input_root, output_root)
