import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch
from MVTecAD2_public_code_utils.mvtec_ad_2_public_offline import MVTecAD2

from src.utils import augment_image, dists2map, plot_ref_images
from src.post_eval import mean_top1p

def run_anomaly_detection(
        model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        save_examples = False,
        masking = None,
        mask_ref_images = False,
        rotation = False,
        knn_metric = 'L2_normalized',
        knn_neighbors = 1,
        faiss_on_cpu = False,
        seed = 0,
        save_patch_dists = True,
        save_tiffs = False,
        dataset="MVTec"):
    """
    Main function to evaluate the anomaly detection performance of a given object/product.

    Parameters:
    - model: The backbone model for feature extraction (and, in case of DINOv2, masking).
    - object_name: The name of the object/product to evaluate.
    - data_root: The root directory of the dataset.
    - n_ref_samples: The number of reference samples to use for evaluation (k-shot). Set to -1 for full-shot setting.
    - object_anomalies: The anomaly types for each object/product.
    - plots_dir: The directory to save the example plots.
    - save_examples: Whether to save example images and plots. Default is True.
    - masking: Whether to apply DINOv2 to estimate the foreground mask (and discard background patches).
    - rotation: Whether to augment reference samples with rotation.
    - knn_metric: The metric to use for kNN search. Default is 'L2_normalized' (1 - cosine similarity)
    - knn_neighbors: The number of nearest neighbors to consider. Default is 1.
    - seed: The seed value for deterministic sampling in few-shot setting. Default is 0.
    - save_patch_dists: Whether to save the patch distances. Default is True. Required to eval detection.
    - save_tiffs: Whether to save the anomaly maps as TIFF files. Default is False. Required to eval segmentation.
    - dataset: Name of the dataset ("MVTec", "VisA", or "MVTec2"). Used to determine directory structure.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    
    # add 'good' to the anomaly types
    type_anomalies = object_anomalies[object_name]
    type_anomalies.append('good')

    # ensure that each type is only evaluated once
    type_anomalies = list(set(type_anomalies))

    # Extract reference features
    features_ref = []
    images_ref = []
    masks_ref = []
    vis_backgroud = []

    test_split = "test_public" if dataset == "MVTec2" else "test"
    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    if dataset == "MVTec2":
        dataset_train = MVTecAD2(object_name, "train")
        img_ref_paths = dataset_train.image_paths
        if n_ref_samples == -1:
            img_ref_samples = img_ref_paths
        else:
            img_ref_samples = img_ref_paths[seed*n_ref_samples:(seed + 1)*n_ref_samples]
    else:
        if n_ref_samples == -1:
            # full-shot setting
            img_ref_samples = [os.path.join(img_ref_folder, f) for f in sorted(os.listdir(img_ref_folder))]
        else:
            # few-shot setting, pick samples in deterministic fashion according to seed
            img_ref_samples = [os.path.join(img_ref_folder, f) for f in sorted(os.listdir(img_ref_folder))[seed*n_ref_samples:(seed + 1)*n_ref_samples]]

    print(f"img_ref_samples : {len(img_ref_samples)}")
    
    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
    
    with torch.inference_mode():
        # start measuring time (feature extraction/memory bank set up)
        start_time = time.time()
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank", leave=False):
            # load reference image...
            img_ref = img_ref_n
            image_ref = cv2.cvtColor(cv2.imread(img_ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # augment reference image (if applicable)... 
            # 總共有 8個方向
            if rotation:
                img_augmented = augment_image(image_ref)
            else:
                img_augmented = [image_ref]
            for i in range(len(img_augmented)):
                image_ref = img_augmented[i]
                image_ref_tensor, grid_size1 = model.prepare_image(image_ref)
                features_ref_i = model.extract_features(image_ref_tensor)
                
                # compute background mask and discard background patches
                mask_ref = model.compute_background_mask(features_ref_i, grid_size1, threshold=10, masking_type=(mask_ref_images and masking))
                features_ref.append(features_ref_i[mask_ref])
                if save_examples:
                    images_ref.append(image_ref)
                    vis_image_background = model.get_embedding_visualization(features_ref_i, grid_size1, mask_ref)
                    masks_ref.append(mask_ref)
                    vis_backgroud.append(vis_image_background)
        
        features_ref = np.concatenate(features_ref, axis=0).astype('float32')

        if faiss_on_cpu:
            # similariy search on CPU
            knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            # similariy search on GPU
            res = faiss.StandardGpuResources()
            knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
            # knn_index = faiss.IndexFlatL2(features_ref.shape[1])
            # knn_index = faiss.index_cpu_to_gpu(res, int(model.device[-1]), knn_index)


        if knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
        knn_index.add(features_ref)

        # end measuring time (for memory bank set up; in seconds, same for all test samples of this object)
        time_memorybank = time.time() - start_time

        # 以上是在建立 memory bank -----------------------------------------------------------------------------------------
        
        # plot some reference samples for inspection
        if save_examples:
            plots_dir_ = f"{plots_dir}/{object_name}/"
            plot_ref_images(images_ref, masks_ref, vis_backgroud, grid_size1, plots_dir_, title = "Reference Images", img_names = [os.path.basename(p) for p in img_ref_samples])  
        
        inference_times = {}
        anomaly_scores = {}

        idx = 0
        # Evaluate anomalies for each anomaly type (and "good")
        for type_anomaly in tqdm(type_anomalies, desc = f"processing test samples ({object_name})"):
            if dataset == "MVTec2":
                dataset_test = MVTecAD2(object_name, "test_public")
                test_paths = [p for p in dataset_test.image_paths if f"/{type_anomaly}/" in p]
            else:
                data_dir = f"{data_root}/{object_name}/{test_split}/{type_anomaly}"
                test_paths = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
            
            if save_patch_dists or save_tiffs:
                os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/{test_split}/{type_anomaly}", exist_ok=True)

            for idx, image_test_path in enumerate(test_paths):
                # start measuring time (inference)
                start_time = time.time()
                img_test_nr = os.path.basename(image_test_path)

                # Extract test features
                image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image_tensor2, grid_size2 = model.prepare_image(image_test)
                features2 = model.extract_features(image_tensor2)

                # Compute background mask (根據不同類別，會使用 masking ， masking 就是區分前景以及背景)
                if masking:
                    mask2 = model.compute_background_mask(features2, grid_size2, threshold=10, masking_type=masking)
                else:
                    mask2 = np.ones(features2.shape[0], dtype=bool)
                if save_examples and idx < 3:
                    vis_image_test_background = model.get_embedding_visualization(features2, grid_size2, mask2)

                # Discard irrelevant features
                features2 = features2[mask2]

                # Compute distances to nearest neighbors in M
                '''
                distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                是查詢的鄰居數量，表示在 FAISS 索引中查找 k 個最近的鄰居。
                distances 代表返回與每個 patch 特徵最接近的 k 個 reference 特徵的 距離。
                '''
                if knn_metric == "L2":
                    distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = np.sqrt(distances)

                elif knn_metric == "L2_normalized":
                    faiss.normalize_L2(features2) 
                    distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = distances / 2   # equivalent to cosine distance (1 - cosine similarity)

                '''
                這段代碼的目的是將計算出的距離結果（distances）應用於背景遮罩（mask2），
                並將結果重新形狀化以符合圖像的網格大小（grid_size2）。具體來說，這些操作的目的通常是在進行異常檢測時，
                將每個圖像的 patch 距離 與背景區域的遮罩結合，並重構為與原始圖像對應的格式。
                
                你在這段代碼中已經計算出了每個 patch 的 anomaly score，這些分數表示每個 patch 的異常程度。
                通過 reshape，你將這些分數重新塑形為與圖像網格大小一致的二維數組（d_masked）。這樣，你就可以將每個 patch 的異常分數映射回圖像的結構中，使其與原始圖像對應。
                '''
                output_distances = np.zeros_like(mask2, dtype=float)
                output_distances[mask2] = distances.squeeze()
                d_masked = output_distances.reshape(grid_size2)
                
                # save inference time
                torch.cuda.synchronize() # Synchronize CUDA kernels before measuring time
                inf_time = time.time() - start_time
                inference_times[f"{type_anomaly}/{img_test_nr}"] = inf_time
                # 拿前面 1% patch 的 anomaly score 作為，整張圖片的 anomaly score
                anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = mean_top1p(output_distances.flatten())

                # Save the anomaly maps (raw as .npy or full resolution .tiff files)
                img_test_nr = img_test_nr.split(".")[0]
                if save_tiffs:
                    # 就是這裡就是把 patch score 轉成 pixel score
                    anomaly_map = dists2map(d_masked, image_test.shape)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/{test_split}/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)
                if save_patch_dists:
                    np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/{test_split}/{type_anomaly}/{img_test_nr}.npy", d_masked)

                # Save some example plots (3 per anomaly type)
                if save_examples and idx < 3:

                    fig, (ax1, ax2, ax3, ax4,) = plt.subplots(1, 4, figsize=(18, 4.5))

                    # plot test image, PCA + mask
                    ax1.imshow(image_test)
                    ax2.imshow(vis_image_test_background)  

                    # plot patch distances 
                    d_masked[~mask2.reshape(grid_size2)] = 0.0
                    plt.colorbar(ax3.imshow(d_masked), ax=ax3, fraction=0.12, pad=0.05, orientation="horizontal")
                    
                    # compute image level anomaly score (mean(top 1%) of patches = empirical tail value at risk for quantile 0.99)
                    score_top1p = mean_top1p(distances)
                    ax4.axvline(score_top1p, color='r', linestyle='dashed', linewidth=1, label=round(score_top1p, 2))
                    ax4.legend()
                    ax4.hist(distances.flatten())

                    ax1.axis('off')
                    ax2.axis('off')
                    ax3.axis('off')

                    ax1.title.set_text("Test")
                    ax2.title.set_text("Test (PCA + Mask)")
                    ax3.title.set_text("Patch Distances (1NN)")
                    ax4.title.set_text("Hist of Distances")

                    plt.suptitle(f"Object: {object_name}, Type: {type_anomaly}, img = ...{image_test_path[-20:]}, object patches = {mask2.sum()}/{mask2.size}")

                    plt.tight_layout()
                    plt.savefig(f"{plots_dir}/{object_name}/examples/example_{type_anomaly}_{idx}.png")
                    plt.close()
                    
    # 回傳這個物件所有異常 type 的所有圖片的 anomaly score
    
    return anomaly_scores, time_memorybank, inference_times