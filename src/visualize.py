from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from .utils import get_dataset_info, dists2map

from  matplotlib.colors import LinearSegmentedColormap
neon_violet = (0.5, 0.1, 0.5, 0.4)
neon_yellow = (0.8, 1.0, 0.02, 0.7)
red_gt = (1.0, 0, 0.0, 0.5)
colors = [(1.0, 1, 1.0, 0.0),  neon_violet, neon_yellow]
cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)


def get_test_gt_map(object_name, anomaly_type, img_nr, experiment, data_root, dataset = "MVTec", good=False):
    """
    Return test sample, ground truth (if not a good sample) and anomaly maps for given experiment and img_nr.
    """ 
    # test sample
    
    if dataset == "MVTec2":
        img_test_path = f"{data_root}/{object_name}/test_public/{anomaly_type}/{img_nr}"
    else:
        img_test_path = f"{data_root}/{object_name}/test/{anomaly_type}/{img_nr}"
    image_test = cv2.cvtColor(cv2.imread(img_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    img_nr = img_nr.split(".")[0]

    # ground truth
    if not good:
        if dataset == "MVTec2":
            gt_path = f"{data_root}/{object_name}/test_public/ground_truth/{anomaly_type}/{img_nr}" + ("_mask.png" if dataset == "MVTec2" else ".png")
        else:
            gt_path = f"{data_root}/{object_name}/ground_truth/{anomaly_type}/{img_nr}" + ("_mask.png" if dataset == "MVTec" else ".png")
        
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # load patch distances for test sample
    if dataset == "MVTec2":
        dists = np.load(f"{experiment}/{object_name}/test_public/{anomaly_type}/{img_nr}.npy")
    else:
        dists = np.load(f"{experiment}/{object_name}/test/{anomaly_type}/{img_nr}.npy")
        
    # anomaly maps
    anomaly_map = dists2map(dists, image_test.shape)
    if good:
        return image_test, anomaly_map
    else:
      return image_test, gt_mask, anomaly_map


def plot_sample(image_test, anomaly_map, axs, cmap, vmax):
    axs.imshow(image_test)
    axs.imshow(anomaly_map, cmap=cmap, vmax=vmax)


def infer_vmax(exp_path, objects, dataset):
    vmax = {}
    for object_name in objects:
        current_max = 0
        if dataset == "MVTec2":
            for test_file_good in os.listdir(f"{exp_path}/{object_name}/test_public/good/"):
                if test_file_good.endswith(".npy"):
                    max_score = np.load(f"{exp_path}/{object_name}/test_public/good/{test_file_good}").max()
                    current_max = max(current_max, max_score)
        else:
            for test_file_good in os.listdir(f"{exp_path}/{object_name}/test/good/"):
                if test_file_good.endswith(".npy"):
                    max_score = np.load(f"{exp_path}/{object_name}/test/good/{test_file_good}").max()
                    current_max = max(current_max, max_score)

        vmax[object_name] = current_max * 1.0
    return vmax


def create_sample_plots(experiment_path, anomaly_maps_dir, seed, dataset, data_root):
    # infer objects and anomalies, preprocessing does not matter
    objects, object_anomalies, _, _ = get_dataset_info(dataset, preprocess = "informed")
    # infer vmax for each object
    vmax = infer_vmax(anomaly_maps_dir, objects, dataset)

    for object_name in tqdm(objects, desc="Plot anomaly maps"):
        n = len(object_anomalies[object_name])
        fig, axs = plt.subplots(n + 1, 5, figsize=(2 * 5, 2* (n + 1)))

        for i, anomaly_type in enumerate(object_anomalies[object_name]):
            # plot five test samples with anomaly maps
            if dataset == "MVTec2":
                first_five_samples = sorted(os.listdir(f"{data_root}/{object_name}/test_public/{anomaly_type}/"))[:5]
            else:
                first_five_samples = sorted(os.listdir(f"{data_root}/{object_name}/test/{anomaly_type}/"))[:5]
            for j, img_nr in enumerate(first_five_samples):
                image_test, gt_mask, anomaly_map = get_test_gt_map(object_name, anomaly_type,
                                                                    img_nr, anomaly_maps_dir, dataset = dataset, data_root = data_root)
                plot_sample(image_test, anomaly_map, axs[i, j], cmap=cmap, vmax=vmax[object_name])
                axs[i, j].axis('off')
                if j == 2:
                    axs[i, j].set_title(f"anomaly type: {anomaly_type}")
        if dataset == "MVTec2":
            first_five_good_samples = sorted(os.listdir(f"{data_root}/{object_name}/test_public/good/"))[:5]
        else:
            first_five_good_samples = sorted(os.listdir(f"{data_root}/{object_name}/test/good/"))[:5]
        for j, img_nr in enumerate(first_five_good_samples):
            # plot five good test samples with anomaly maps for comparison
            image_test, anomaly_map = get_test_gt_map(object_name, "good", img_nr, 
                                                      anomaly_maps_dir, dataset = dataset, data_root = data_root, good=True)
            axs[n, j].imshow(image_test)
            axs[n, j].imshow(anomaly_map, cmap=cmap, vmax=vmax[object_name])
            axs[n, j].axis('off')
            if j == 2:
                axs[n, j].set_title(f"good test samples (for comparison)")

        plt.tight_layout()
        plt.savefig(f"{experiment_path}/{object_name}/anomaly_maps_examples_seed={seed}.png")
        plt.close()

def create_heat_map(experiment_path, anomaly_maps_dir, seed, dataset, data_root):
    """Create heat map overlays of ground truth masks on test images.

    Parameters
    ----------
    dataset : str
        Name of the dataset (e.g. "MVTec", "MVTec2").
    data_root : str (--data_root data/mvtec_ad_2)
        Root path of the dataset.
    output_dir : str
        Directory where the overlay images will be saved.
    """
    
    print(f"=========== Save Heat_Map ===========")
    objects, object_anomalies, _, _ = get_dataset_info(dataset, preprocess="informed")

    for object_name in tqdm(objects, desc="Create heat maps"):
        for anomaly_type in object_anomalies[object_name]:
            if dataset == "MVTec2":
                img_dir = os.path.join(data_root, object_name, "test_public", anomaly_type)
                gt_dir = os.path.join(data_root, object_name, "test_public", "ground_truth", anomaly_type)
            else:
                img_dir = os.path.join(data_root, object_name, "test", anomaly_type)
                gt_dir = os.path.join(data_root, object_name, "ground_truth", anomaly_type)

            if not os.path.exists(img_dir):
                continue

            save_path = f"{experiment_path}/{object_name}/heat_map/{anomaly_type}"
            os.makedirs(save_path, exist_ok=True)

            for img_name in sorted(os.listdir(img_dir)):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    continue
                img_path = os.path.join(img_dir, img_name)
                mask_name = os.path.splitext(img_name)[0] + "_mask.png"
                mask_path = os.path.join(gt_dir, mask_name)
                if not os.path.exists(mask_path):
                    continue

                # === Load 原圖與遮罩 ===
                img = Image.open(img_path).convert("RGBA")
                mask = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask)

                # === 建立紅色 overlay，alpha 根據 mask ===
                overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
                overlay_arr = np.array(overlay)
                overlay_arr[..., 3] = (mask_arr > 0) * 128
                overlay = Image.fromarray(overlay_arr, mode="RGBA")

                # === 疊圖 ===
                heat_img = Image.alpha_composite(img, overlay).convert("RGB")
                
                # === 儲存圖片 ===
                save_img_path = os.path.join(save_path, img_name)
                Image.fromarray(heat_np).save(save_img_path)