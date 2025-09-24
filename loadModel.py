from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger
from recbole.data.interaction import Interaction
from recbole.data.dataset import Dataset
from recbole.data.dataloader import FullSortEvalDataLoader

np.bool8 = np.bool
np.float_ = np.float64
np.complex_ = np.complex128
np.unicode_ = np.str_

config_dict = {
    'epochs': 60,
    'stopping_step': 2,
    'eval_step': 1,
    'learning_rate': 0.0004,
    'mlp-hidden-size': '[128,64,32,16]',
    'num_conv_kernel': 128,
    'embedding_size': 64,
    'train_batch_size': 2048,
    'eval_batch_size': 4096,
}

config = Config(
    model='DeepFM',
    dataset='ml-100k',
    config_dict=config_dict,
)

device = config['device']

# 1. Load Dataset
dataset = torch.load('saved/DeepFM-2025-09-17_21-51-07/ml-100k_dataset.pth', map_location=device)

# 2. Load DataLoader
data_loader = torch.load('saved/DeepFM-2025-09-17_21-51-07/ml-100k_train_data.pth', map_location=device)

# 3. Load Model
model = DeepFM(config, data_loader.dataset).to(config['device'])
model.load_state_dict(torch.load('saved/DeepFM-2025-09-17_21-51-07/DeepFM_model_weights.pth', map_location=device))

u_idx = dataset.token2id('user_id', ['196'])[0]
i_idx = dataset.token2id('item_id', ['346'])[0]

# Lấy thông tin gốc từ dataset (có đủ age, gender,...)
user_feat = dataset.get_user_feature()[u_idx]
item_feat = dataset.get_item_feature()[i_idx]


base_inter = Interaction({
    'user_id': torch.tensor([u_idx]),
    'item_id': torch.tensor([i_idx]),
})
interaction = dataset.join(base_inter)

model.eval()
with torch.no_grad():
    score = model.predict(interaction)
print(f"Predict score {u_idx} and item {i_idx}: {score.item()}")

#gemini

import torch
import torch.nn.functional as F

def generate_candidates_from_history(interacted_item_ids, model, dataset, device, K=100):
    """
    Sinh ứng viên từ lịch sử tương tác dựa trên embedding của item trong RecBole.

    Args:
        interacted_item_ids (list[str]): danh sách item_id gốc (string).
        model: mô hình RecBole đã huấn luyện (ví dụ DeepFM).
        dataset: Dataset RecBole đã load.
        device: torch.device.
        K (int): số lượng ứng viên muốn lấy.

    Returns:
        list[str]: danh sách item_id ứng viên.
    """
    # --- B1: Lấy embedding toàn cục ---
    full_embedding_table = model.token_embedding_table.embedding.weight  # Tensor [n_tokens, emb_dim]

    # --- B2: Lấy index của toàn bộ item ---
    all_item_indices = dataset.token2id('item_id', dataset.id2token('item_id', list(range(dataset.num('item_id')))))
    all_item_indices_tensor = torch.tensor(all_item_indices, device=device, dtype=torch.long)

    # --- B3: Lấy embedding cho toàn bộ item ---
    item_embeddings = full_embedding_table[all_item_indices_tensor]

    # --- B4: Lấy embedding của item người dùng đã tương tác ---
    interacted_indices = dataset.token2id('item_id', interacted_item_ids)
    interacted_indices_tensor = torch.tensor(interacted_indices, device=device, dtype=torch.long)
    interacted_embeddings = full_embedding_table[interacted_indices_tensor]

    # --- B5: Tính cosine similarity ---
    similarity_scores = F.cosine_similarity(
        interacted_embeddings.unsqueeze(1),   # [num_interacted, 1, dim]
        item_embeddings.unsqueeze(0),         # [1, num_items, dim]
        dim=2
    )
    top_scores_per_item, _ = torch.max(similarity_scores, dim=0)

    # --- B6: Loại bỏ item đã xem ---
    top_scores_per_item[interacted_indices_tensor] = -1.0

    # --- B7: Chọn top-K ---
    _, top_k_indices = torch.topk(top_scores_per_item, k=K)

    # --- B8: Map ngược về item_id gốc ---
    candidate_ids = dataset.id2token('item_id', top_k_indices.cpu().numpy())

    return candidate_ids

items_he_liked = ["242", "302", "377"]
candidate_items = generate_candidates_from_history(items_he_liked, model, dataset, device=config['device'], K=10)

print(f"Đã tìm thấy {len(candidate_items)} ứng viên.")
print("Một vài ứng viên:", candidate_items)
  
def get_recommendations(list_items, K=10):
    candidate_items = generate_candidates_from_history(list_items, model, dataset, device=config['device'], K=K)
    # Chuyển numpy array thành Python list để có thể serialize JSON
    if isinstance(candidate_items, np.ndarray):
        return candidate_items.tolist()
    return list(candidate_items)

list_items = ["242", "302", "377"]
print(get_recommendations(list_items))