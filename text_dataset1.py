import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class DisentanglementDataset(Dataset):
    def __init__(self, metadata_csv, feature_dir):
        metadata = pd.read_csv(metadata_csv)
        self.feature_dir = feature_dir

        print("Preprocessing and preloading all data for 2-way disentanglement...")
        self.texts = []
        self.a_vectors = []
        self.m_vectors = []

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Preloading data"):
            text_full = row['text']
            relative_path_with_ext = row['file_name'].strip()
            path_without_ext, _ = os.path.splitext(relative_path_with_ext)

            anatomy_path = os.path.join(self.feature_dir, path_without_ext + '_anatomy.npy')
            modality_path = os.path.join(self.feature_dir, path_without_ext + '_modality.npy')  # <-- 新路径

            if not all(os.path.exists(p) for p in [anatomy_path, modality_path]):
                print(f"\nWarning: Feature files not found for '{relative_path_with_ext}'. Skipping.")
                continue

            a_img_map = torch.from_numpy(np.load(anatomy_path))
            a_img_vector = torch.mean(a_img_map, dim=(1, 2))

            m_img_vector = torch.from_numpy(np.load(modality_path))

            self.texts.append(text_full)
            self.a_vectors.append(a_img_vector)
            self.m_vectors.append(m_img_vector)

        print(f"Dataset preloaded successfully. Total valid samples: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.a_vectors[idx], self.m_vectors[idx]