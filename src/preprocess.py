from dataset import LiDARDatasetProcessor, save_pt_splits

if __name__ == "__main__":
    processor = LiDARDatasetProcessor(
        csv_file="data/raw/Lidar_dataset_original_4scales.csv",
        max_points=512,
    )

    X, y, splits, ids = processor.run()

    save_pt_splits(X, y, splits, ids)