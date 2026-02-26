import os
import numpy as np
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm

from .data import Data
from utils import data_utils


class HAM10000Loader:

    def __init__(self, data_root, input_shape=(224, 224)):
        self.data_root = data_root
        self.input_shape = input_shape
        self.cache_dir = os.path.join(self.data_root, 'preprocessed_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        print("HAM10000Loader initialized.")

    def load_dataset(self, split_type='train'):
        assert split_type in ['train', 'test'], "split_type must be 'train' or 'test'"

        cache_file_path = os.path.join(self.cache_dir, f'ham10000_{split_type}.npz')
        if os.path.exists(cache_file_path):
            print(f"Loading preprocessed '{split_type}' data from cache...")
            cached_data = np.load(cache_file_path)
            return Data(cached_data['images'], cached_data['masks'], cached_data['index'], None)

        print(f"Cache not found. Loading '{split_type}' data from source files...")

        split_path = os.path.join(self.data_root, split_type)
        class_folders = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])

        images_list, masks_list, index_list = [], [], []

        for class_name in class_folders:
            class_path = os.path.join(split_path, class_name)
            image_paths = sorted(glob(os.path.join(class_path, '*.jpg')))

            for img_path in tqdm(image_paths, desc=f"Processing {class_name}"):
                mask_path = img_path.replace('.jpg', '_segmentation.png')

                if os.path.exists(mask_path):
                    image = np.array(Image.open(img_path).convert('RGB'))
                    mask = np.array(Image.open(mask_path).convert('L'))

                    image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)

                    image = data_utils.normalise(image.astype(np.float32), -1, 1)
                    mask = (mask > 128).astype(np.float32)

                    images_list.append(image)
                    masks_list.append(mask)
                    index_list.append(os.path.join(class_name, os.path.basename(img_path)))  # 保存相对路径作为唯一ID

        images = np.transpose(np.stack(images_list, axis=0), (0, 3, 1, 2)).astype(np.float32)
        masks = np.expand_dims(np.stack(masks_list, axis=0), axis=1).astype(np.float32)

        print(f"\nSaving '{split_type}' data to cache...")
        np.savez_compressed(cache_file_path, images=images, masks=masks, index=np.array(index_list))

        return Data(images, masks, np.array(index_list), None)