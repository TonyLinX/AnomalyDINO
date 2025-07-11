.PHONY: all train test evaluate metrics clean

# 預設目標（可以改成你最常用的）
all: train test evaluate metrics

# 執行訓練腳本
train:
	./run_train.sh

# 執行測試腳本
test:
	./runtest.sh

# 執行 anomaly map 評估
evaluate:
	python mvtec_ad_evaluation/evaluate_experiment.py \
		--anomaly_maps_dir mvtec_results/anomaly_images \
		--dataset_base_dir mvtec_ad_2 \
		--output_dir metrics

evaluate_metrics:
	python mvtec_ad_evaluation/evaluate_experiment.py \
		--anomaly_maps_dir mvtec_results_shift_3/anomaly_images \
		--dataset_base_dir mvtec_ad_2_shift_3 \
		--output_dir metrics

	python mvtec_ad_evaluation/print_metrics.py

# 輸出 metrics 結果
metrics:
	python mvtec_ad_evaluation/print_metrics.py

measure:
	python measure_runtime_and_memory.py \
		--config_path  config/fabric.yaml \
		--checkpoint_path results/2025_07_02_19_10_55/checkpoints_45.pkl

biniarize:
	python binarize_anomaly_maps_dynamic.py 

submit:
	python MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py mvtec_results


# 可選：清除輸出資料夾
clean:
	rm -rf metrics mvtec_results
