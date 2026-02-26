import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):

    def __init__(self, model_name='ClinicalBERT', device='cpu'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.embedding_dim = self.model.config.hidden_size

        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        with torch.no_grad():
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.model.device)

            model_output = self.model(**encoded_input)

            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

class TextDisentangler(nn.Module):

    def __init__(self, input_dim, anatomy_dim, modality_dim, hidden_dim=512):
        super().__init__()

        self.anatomy_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, anatomy_dim)
        )

        self.modality_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, modality_dim)
        )

        decoder_input_dim = anatomy_dim + modality_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, E_full):
        t_a = self.anatomy_encoder(E_full)
        t_m = self.modality_encoder(E_full)

        t_combined = torch.cat([t_a, t_m], dim=1)
        E_full_recon = self.decoder(t_combined)

        return t_a, t_m, E_full_recon