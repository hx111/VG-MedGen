import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from models.text_models1 import TextEncoder, TextDisentangler
from text_dataset1 import DisentanglementDataset


def parse_arguments():

    parser = argparse.ArgumentParser(description="2-Way Text Disentangler Training")
    parser.add_argument('--metadata_csv', type=str, required=True, help='Path to the metadata CSV file with text descriptions.')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Path to the root directory of extracted 2-way image features.')  # 更新帮助信息
    parser.add_argument('--save_path', type=str, default='./text_disentangler/',
                        help='Path to save the trained disentangler model.')
    parser.add_argument('--text_encoder_model', type=str, default='',
                        help='Name of the pretrained text encoder model from Hugging Face.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use.')
    parser.add_argument('--align_weight', type=float, default=1.0, help='Similarity loss weight (alpha).')
    parser.add_argument('--recon_weight', type=float, default=0.5, help='Reconstruction loss weight (gamma).')
    return parser.parse_args()


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None
    texts = [item[0] for item in batch]
    tensors = [item[1:] for item in batch]
    collated_tensors = torch.utils.data.default_collate(tensors)
    return [texts] + list(collated_tensors)


if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    dataset = DisentanglementDataset(args.metadata_csv, args.feature_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print("Dataset ready.")

    print("Loading text encoder and initializing disentangler...")
    text_encoder = TextEncoder(model_name=args.text_encoder_model, device=DEVICE)

    anatomy_dim_from_image = 8
    modality_dim_from_image = 8

    text_disentangler = TextDisentangler(
        input_dim=text_encoder.embedding_dim,
        anatomy_dim=anatomy_dim_from_image,
        modality_dim=modality_dim_from_image
    ).to(DEVICE)

    optimizer = optim.Adam(text_disentangler.parameters(), lr=args.lr)
    print("Models ready.")
    loss_align_fn = nn.CosineEmbeddingLoss().to(DEVICE)
    loss_recon_fn = nn.MSELoss().to(DEVICE)

    print("--- Starting Training for Text Disentangler ---")
    for epoch in range(args.epochs):
        total_loss_all_epoch = 0
        total_loss_align_epoch = 0
        total_loss_recon_epoch = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for text_batch, a_img_batch, m_img_batch in pbar:

            if text_batch is None:
                continue
            a_img_batch = a_img_batch.to(DEVICE)
            m_img_batch = m_img_batch.to(DEVICE)

            with torch.no_grad():
                E_full_batch = text_encoder(text_batch)

            t_a, t_m, E_full_recon = text_disentangler(E_full_batch)

            target = torch.ones(len(text_batch), device=DEVICE)
            loss_align_a = loss_align_fn(t_a, a_img_batch.detach(), target)
            loss_align_m = loss_align_fn(t_m, m_img_batch.detach(), target)
            loss_align = loss_align_a + loss_align_m

            loss_recon = loss_recon_fn(E_full_recon, E_full_batch.detach())

            total_loss_batch = args.align_weight * loss_align + \
                               args.recon_weight * loss_recon

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss_all_epoch += total_loss_batch.item()
            total_loss_align_epoch += loss_align.item()
            total_loss_recon_epoch += loss_recon.item()

            pbar.set_postfix(
                loss=f"{total_loss_batch.item():.4f}",
                align=f"{loss_align.item():.4f}",
                recon=f"{loss_recon.item():.4f}"
            )

        num_batches = len(dataloader)
        if num_batches > 0:
            avg_loss = total_loss_all_epoch / num_batches
            avg_align = total_loss_align_epoch / num_batches
            avg_recon = total_loss_recon_epoch / num_batches
            print(f"\nEpoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f} "
                  f"(Align: {avg_align:.4f}, Recon: {avg_recon:.4f})")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            os.makedirs(args.save_path, exist_ok=True)
            save_file = os.path.join(args.save_path, f'disentangler_2way_epoch_{epoch + 1}.pth')
            torch.save(text_disentangler.state_dict(), save_file)
            print(f"Model checkpoint saved to {save_file}\n")

    print("--- Training Finished ---")