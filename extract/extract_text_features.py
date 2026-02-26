import torch
import numpy as np
import argparse
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from models.text_models1 import TextEncoder, TextDisentangler
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from models.text_models1 import TextEncoder, TextDisentangler

class TextMetadataDataset(Dataset):
    def __init__(self, metadata_csv_path):
        print(f"Reading metadata from: {metadata_csv_path}")
        self.metadata = pd.read_csv(metadata_csv_path)
        print(f"Found {len(self.metadata)} text samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        text = row['text']
        relative_path = row['file_name'].strip()
        return text, relative_path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch extract disentangled features from all texts in a metadata file.")

    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to the metadata CSV file containing text and file_name columns.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Root directory to save the extracted text features.')
    parser.add_argument('--disentangler_weights_path', type=str, required=True,
                        help='Path to the trained 2-way TextDisentangler .pth file.')
    parser.add_argument('--bert_model_name', type=str, default='',
                        help='Name or path of the pre-trained BERT model.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. Use "cpu" for CPU.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.gpu.lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')
    print(f"Using device: {device}")

    print("Loading models...")
    text_encoder = TextEncoder(model_name=args.bert_model_name, device=device)

    anatomy_dim = 8
    modality_dim = 8
    text_disentangler = TextDisentangler(
        input_dim=text_encoder.embedding_dim,
        anatomy_dim=anatomy_dim,
        modality_dim=modality_dim
    ).to(device)

    text_disentangler.load_state_dict(torch.load(args.disentangler_weights_path, map_location=device))
    text_disentangler.eval()
    print("Models loaded successfully.")

    dataset = TextMetadataDataset(args.metadata_csv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"--- Starting feature extraction. Saving to: {args.save_path} ---")

    with torch.no_grad():
        for texts_batch, relative_paths_batch in tqdm(dataloader, desc="Extracting text features"):

            E_full_batch = text_encoder(texts_batch)
            t_a_batch, t_m_batch, _ = text_disentangler(E_full_batch)

            t_a_batch_np = t_a_batch.cpu().numpy()
            t_m_batch_np = t_m_batch.cpu().numpy()

            for i in range(len(texts_batch)):
                relative_path = relative_paths_batch[i]
                class_name = os.path.dirname(relative_path)
                base_filename = os.path.basename(relative_path)
                clean_filename, _ = os.path.splitext(base_filename)

                class_save_dir = os.path.join(args.save_path, class_name)
                os.makedirs(class_save_dir, exist_ok=True)

                anatomy_save_path = os.path.join(class_save_dir, f"{clean_filename}_text_anatomy.npy")
                modality_save_path = os.path.join(class_save_dir, f"{clean_filename}_text_modality.npy")

                np.save(anatomy_save_path, t_a_batch_np[i])
                np.save(modality_save_path, t_m_batch_np[i])

    print("\n--- Batch Feature Extraction Complete! ---")
    print(f"All text features have been saved to: {args.save_path}")