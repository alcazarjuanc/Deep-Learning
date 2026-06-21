from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class POIDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes, metadatos y etiquetas de POIs.
    """

    def __init__(self, dataframe, metadata_array, labels, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.metadata_array = metadata_array.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = Path(row["image_path_local"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        metadata = torch.tensor(self.metadata_array[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "image": image,
            "metadata": metadata,
            "label": label
        }