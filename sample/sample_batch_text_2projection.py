import os
import torch
import pandas as pd
import numpy as np
from diffusers import PixArtAlphaPipeline
from peft import PeftModel
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MultiFeatureProjector(nn.Module):

    def __init__(self, anatomy_dim=8, modality_dim=8, output_dim=4096, hidden_dim=256):
        super().__init__()
        self.norm_a = nn.LayerNorm(anatomy_dim, eps=1e-6)
        self.norm_m = nn.LayerNorm(modality_dim, eps=1e-6)

        def create_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )

        self.anatomy_proj = create_mlp(anatomy_dim)
        self.modality_proj = create_mlp(modality_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.type_embeddings = nn.Parameter(torch.randn(2, output_dim) * 0.02)

    def forward(self, anatomy_feat, modality_feat):
        original_dtype = anatomy_feat.dtype

        anatomy_feat = self.norm_a(anatomy_feat.float()).to(original_dtype)
        modality_feat = self.norm_m(modality_feat.float()).to(original_dtype)

        a_proj = self.anatomy_proj(anatomy_feat)
        m_proj = self.modality_proj(modality_feat)

        a_proj = a_proj.unsqueeze(1)
        m_proj = m_proj.unsqueeze(1)

        a_final = a_proj + self.type_embeddings[0]
        m_final = m_proj + self.type_embeddings[1]

        feature_sequence = torch.cat([a_final, m_final], dim=1)
        return feature_sequence

class FeatureDataset(Dataset):
    def __init__(self, df, features_base_dir):
        self.df = df
        self.features_base_dir = features_base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path_info = row['file_name']
        class_label, image_filename = os.path.split(file_path_info)
        image_id = os.path.splitext(image_filename)[0]

        anatomy_path = os.path.join(self.features_base_dir, class_label, f"{image_id}_text_anatomy.npy")
        modality_path = os.path.join(self.features_base_dir, class_label, f"{image_id}_text_modality.npy")

        if not all(map(os.path.exists, [anatomy_path, modality_path])):
            print(f"Warning: Features for '{image_id}' not found. Returning None.")
            return None

        anatomy_feat = torch.from_numpy(np.load(anatomy_path)).squeeze()
        modality_feat = torch.from_numpy(np.load(modality_path)).squeeze()

        return {"anatomy_features": anatomy_feat, "modality_features": modality_feat,
                "class_label": class_label, "image_id": image_id}


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    anatomy_features = torch.stack([item["anatomy_features"] for item in batch])
    modality_features = torch.stack([item["modality_features"] for item in batch])
    class_labels = [item["class_label"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    return {"anatomy_features": anatomy_features, "modality_features": modality_features,
            "class_label": class_labels, "image_id": image_ids}


def main():
    class Args:
        base_model_id = "PixArt-alphaPixArt-XL-2-512x512"
        weights_dir = "../pixart_alphaPixArt_XL_LoRA/text_projection1"
        metadata_csv_path = "../dataset/HAM10000/train/metadata.csv"
        features_dir = "../dataset/HAM10000/extracted_text_features"
        output_dir = "../dataset/synthetic_HAM10000_from_2projection1"
        batch_size = 8
        num_inference_steps = 50
        guidance_scale = 4.5

    args = Args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"Using device: {device}, dtype: {dtype}")

    print("Step 1: Loading models...")
    feature_projector = MultiFeatureProjector()
    projector_path = os.path.join(args.weights_dir, "feature_projector.pth")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"feature_projector.pth not found in {args.weights_dir}")
    feature_projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    feature_projector.to(device, dtype=dtype)
    feature_projector.norm_a.to(torch.float32)
    feature_projector.norm_m.to(torch.float32)
    feature_projector.eval()
    print("  - MultiFeatureProjector loaded and dtypes corrected.")

    pipe = PixArtAlphaPipeline.from_pretrained(args.base_model_id, torch_dtype=dtype)
    print("  - Base PixArtAlphaPipeline loaded.")

    lora_path = os.path.join(args.weights_dir, "transformer_lora")
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"transformer_lora directory not found in {args.weights_dir}")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
    pipe.transformer = pipe.transformer.merge_and_unload()
    print("  - LoRA weights loaded and merged into Transformer.")

    pipe.to(device)
    print("All models loaded successfully.\n")

    print(f"Step 2: Preparing data from '{args.metadata_csv_path}'...")
    df = pd.read_csv(args.metadata_csv_path)
    dataset = FeatureDataset(df, args.features_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Found {len(df)} entries to process.\n")

    print("Step 3: Starting image generation...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Batches"):
            if batch is None: continue

            anatomy = batch["anatomy_features"].to(device, dtype=dtype)
            modality = batch["modality_features"].to(device, dtype=dtype)

            prompt_embeds = feature_projector(anatomy, modality)
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            prompt_attention_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=dtype)
            negative_prompt_attention_mask = torch.ones_like(prompt_attention_mask)

            images = pipe(
                prompt=None, negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images

            for i, image in enumerate(images):
                class_label = batch["class_label"][i]
                image_id = batch["image_id"][i]
                output_class_dir = os.path.join(args.output_dir, class_label)
                os.makedirs(output_class_dir, exist_ok=True)

                save_path = os.path.join(output_class_dir, f"{image_id}.png")
                image.save(save_path)

    print(f"\nGeneration complete! Images are saved in '{args.output_dir}'")


if __name__ == "__main__":
    main()
