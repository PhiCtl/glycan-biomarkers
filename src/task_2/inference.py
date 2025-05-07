
import pandas as pd
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from glycowork.ml.models import SweetNet
from glycowork.ml.inference import glycans_to_emb
from transformers import RobertaForMaskedLM

from src.task_2.datasets import GlycanDataset


def get_embeddings(data: pd.DataFrame, model, tokenizer=None, save_path=None) -> pd.DataFrame: 



    assert('glycan' in data.columns)

    if isinstance(model, SweetNet):
        
        embeddings = glycans_to_emb(data['glycan'].values, model)
    
    elif isinstance(model, RobertaForMaskedLM):

        assert(tokenizer is not None)
        dataset = GlycanDataset(data['glycan'].tolist(),
                                tokenizer=tokenizer, max_length=510) # TODO errors with a few long glycans
        data_loader = DataLoader(dataset=dataset, batch_size = 64, shuffle=False)
        embeddings = []

        for batch in tqdm(data_loader):
            with torch.no_grad():
                embed = model(**batch)
            last_hidden_state = embed.hidden_states[-1].cpu().numpy()
            attention_mask = batch['attention_mask'].unsqueeze(-1).numpy()
            # Average token embedding to build sequence embedding
            sum_embedding = (last_hidden_state*attention_mask).sum(axis=1).squeeze()
            sum_mask = (attention_mask.sum(axis=1))
            embeddings.append(sum_embedding / sum_mask)
        
        embeddings = pd.DataFrame(np.concatenate(embeddings, axis=0))
    
    else :
        return
    
    embeddings['glycan'] = data['glycan'].tolist()

    if save_path:
        dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(save_path, f'embeddings_{dt}.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)
        
    return embeddings
