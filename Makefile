.PHONY: test_MVTec

test_MVTec_448:
	python run_anomalydino.py --dataset MVTec --resolution 448 --fpr 0.30 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_anomaly_detection 
	python run_anomalydino.py --dataset MVTec --resolution 448 --fpr 0.05 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_anomaly_detection

test_MVTec_672:
	python run_anomalydino.py --dataset MVTec --resolution 672 --fpr 0.30 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_anomaly_detection 
	python run_anomalydino.py --dataset MVTec --resolution 672 --fpr 0.05 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_anomaly_detection

test_MVTec_0:
	python run_anomalydino.py --dataset MVTec --resolution 0 --fpr 0.30 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_anomaly_detection 
	python run_anomalydino.py --dataset MVTec --resolution 0 --fpr 0.05 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_anomaly_detection

test_MVTec2_448:
	python run_anomalydino.py --dataset MVTec2 --resolution 448 --fpr 0.30 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_ad_2 

test_MVTec2:
	python run_anomalydino.py --dataset MVTec2 --resolution 0 --fpr 0.05 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_ad_2 
	python run_anomalydino.py --dataset MVTec2 --resolution 448 --fpr 0.05 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_ad_2 
	python run_anomalydino.py --dataset MVTec2 --resolution 672 --fpr 0.05 --shots 1 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_ad_2 

test_MVTec2_16shot_k8:
	python run_anomalydino.py --dataset MVTec2 --resolution 0 --fpr 0.05 --shots 16 --k_neighbors 8 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_ad_2

test_MVTec2_16shot:
	python run_anomalydino.py --dataset MVTec2 --resolution 0 --fpr 0.05 --shots 16 --num_seeds 1 --preprocess agnostic --data_root data/mvtec_ad_2
	
debug:
	python run_anomalydino.py --dataset MVTec2 --resolution 224 --fpr 0.30 --shots 1 --num_seeds 1 --preprocess informed --data_root data/mvtec_ad_2 --debug True
