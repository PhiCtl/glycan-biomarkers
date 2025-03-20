from datetime import datetime
import os
import pickle
import pandas as pd

import torch
from tqdm import tqdm
from transformers import pipeline, RobertaForMaskedLM, RobertaConfig
from glycowork.ml.models import SweetNet
from glycowork.ml.inference import glycans_to_emb

import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, path=None, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.path is not None:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def load_file(path, class_=None):
    # Extract file extension (lowercase for consistency)
    ext = os.path.splitext(path)[1].lower()
    
    # Dictionary to map extensions to appropriate pandas functions
    loaders = {
        '.csv': pd.read_csv,
        '.pkl': pd.read_pickle,
        '.json': pd.read_json,
        '.parquet': pd.read_parquet,
    }
    
    # Load the file using the right function
    if ext in loaders:
        data =  loaders[ext](path)
        if class_:
            return data.explode(class_)
        else:
            return data
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_model(path, model_type: str):
        
    if model_type == 'SweetNet':

        model_ft = torch.load(path)
        
    elif model_type == 'RoBERTa':

        config = RobertaConfig.from_pretrained(path)
        config.output_hidden_states = True
        model_ft = RobertaForMaskedLM.from_pretrained(path, config=config)
    
    else:
        raise NotImplementedError()

    model_ft.eval()
    return model_ft


def get_embeddings(data: pd.DataFrame, model, tokenizer=None, save_path=None):

    assert('glycan' in data.columns)

    if isinstance(model, SweetNet):
        embeddings = glycans_to_emb(data['glycan'].values, model)
        if save_path:
            dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            with open(os.path.join(save_path, f'embeddings_{dt}.pkl'), 'wb') as f:
                pickle.dump(embeddings, f)
        return embeddings
    
    elif isinstance(model, RobertaForMaskedLM):

        assert(tokenizer is not None)

        errors_g, embeddings = [], []
        for g in tqdm(data['glycan'].values):
            encoded_glycan = tokenizer(g, return_tensors='pt', padding=True, truncation=True)
            try : 
                with torch.no_grad():
                    embed = model(**encoded_glycan)
                last_hidden_state = embed[-1][-1]
                # Average token embedding to build sequence embedding
                glycan_embed = last_hidden_state.squeeze(0).mean(dim=0).numpy()
                embeddings.append(glycan_embed)
            except Exception as e:
                print(f"Error with {g} : {e}")
                errors_g.append(g)
        
        if save_path:
            dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            with open(os.path.join(save_path, f'embeddings_{dt}.pkl'), 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings, errors_g
    else:
        raise NotImplementedError()
    
    
    
