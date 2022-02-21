from tensorflow.keras.preprocessing import image_dataset_from_directory


class DatasetLoader:
    def __init__(self):
        pass

    def build_dataset(data_dir,
                      image_size,
                      batch_size):
        dataset = image_dataset_from_directory(
            data_dir, label_mode=None, image_size=image_size,
            batch_size=batch_size
        )
        return dataset

    def normalize_dataset(dataset):
        # Normalizing to -1,1
        return dataset.map(lambda x: (x - 127.5) / 127.5)
