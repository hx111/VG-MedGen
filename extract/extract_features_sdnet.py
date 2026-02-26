
import torch
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

current_script_path = os.path.abspath(__file__)

current_dir = os.path.dirname(current_script_path)

project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from loaders.ham_loader import HAM10000Loader
    import models
except ImportError as e:
    print(f"Failed to import project modules. Error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted to add project root: {project_root}")
    sys.exit(1)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Feature Extraction for SDNet (saved by directory structure)")

    parser.add_argument('--load_weights_path', type=str, required=True,
                        help='Path to the trained SDNet checkpoint (.pth file).')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the HAM10000 dataset root directory.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the extracted features.')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID to use.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for faster inference.')

    return parser.parse_args(args)

class FeatureExtractionDataset(Dataset):
    def __init__(self, data_source):
        self.images = data_source.images
        self.image_paths = data_source.index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        path = self.image_paths[idx]
        return image, path


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting 2-Level Feature Extraction (by directory structure) ---")
    print(f"Using device: {device}")

    print(f"Loading model checkpoint from: {args.load_weights_path}")
    checkpoint = torch.load(args.load_weights_path, map_location=device)

    print("Reconstructing model parameters from checkpoint...")
    try:
        model_params = {
            'width': checkpoint['width'], 'height': checkpoint['height'], 'ndf': checkpoint['ndf'],
            'norm': checkpoint['norm'], 'upsample': checkpoint['upsample'], 'num_classes': checkpoint['num_classes'],
            'anatomy_out_channels': checkpoint['anatomy_out_channels'], 'z_length': checkpoint['z_length'],
            'num_mask_channels': checkpoint['num_mask_channels'],
        }
    except KeyError as e:
        print(f"Error: A required model parameter is missing from the checkpoint file: {e}")
        sys.exit(1)

    model = models.get_model('sdnet', model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    print("Loading HAM10000 dataset for feature extraction...")
    loader = HAM10000Loader(args.data_path, input_shape=(model_params['width'], model_params['height']))

    dataset_np = loader.load_dataset(split_type='train')

    if not hasattr(dataset_np, 'index'):
        print("Error: The loaded dataset object does not have an 'index' attribute.")
        print(
            "Please ensure your HAM10000Loader returns a Data object with the 'index' attribute containing relative paths.")
        sys.exit(1)

    feature_dataset = FeatureExtractionDataset(dataset_np)
    feature_loader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Data loaded: {len(feature_dataset)} samples.")

    print(f"--- Extracting and saving features to: {args.save_path} ---")

    with torch.no_grad():
        for b_images, b_paths in tqdm(feature_loader, desc="Extracting Features"):
            b_images = b_images.to(device)

            outputs = model(b_images, None, 'test')
            anatomy_features = outputs[4]
            modality_features = outputs[6]

            anatomy_np = anatomy_features.cpu().numpy()
            modality_np = modality_features.cpu().numpy()

            for i in range(b_images.size(0)):
                relative_path = b_paths[i]

                class_name = os.path.dirname(relative_path)

                if not class_name:
                    class_name = "unclassified"

                base_filename = os.path.basename(relative_path)
                clean_filename, _ = os.path.splitext(base_filename)

                class_save_dir = os.path.join(args.save_path, class_name)
                os.makedirs(class_save_dir, exist_ok=True)

                np.save(os.path.join(class_save_dir, f"{clean_filename}_anatomy.npy"), anatomy_np[i])
                np.save(os.path.join(class_save_dir, f"{clean_filename}_modality.npy"), modality_np[i])

    print("\n--- Feature Extraction Complete! ---")