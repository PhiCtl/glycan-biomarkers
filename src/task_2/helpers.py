from datetime import datetime
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
#from tqdm import tqdm
#from transformers import pipeline, RobertaForMaskedLM, RobertaConfig
#from glycowork.ml.models import SweetNet
#from glycowork.ml.inference import glycans_to_emb

import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model(path, model_type: str, config: dict):
        
    if model_type == 'SweetNet':
        import torch
        model_ft = torch.load(path, map_location=DEVICE, weights_only=False)
        
    elif model_type == 'RoBERTa':
        from transformers import RobertaForMaskedLM, RobertaConfig
        model_config = RobertaConfig.from_pretrained(config['training']['output_dir'])
        model_config.output_hidden_states = True
        model_config.vocab_size = config['tokenizer']['vocab_size']
        model_config.max_position_embeddings = config['model']['max_position_embeddings']
        model_ft = RobertaForMaskedLM.from_pretrained(config['training']['output_dir'], config=model_config)
    
    else:
        raise NotImplementedError()

    model_ft.eval()
    return model_ft


def plot_embeddings(embed:np.ndarray, data:pd.DataFrame, hue:str, limit:int = 5, errors=None, seed=42):

    from sklearn.manifold import TSNE

    assert(hue in data.columns)
    assert(embed.shape[0] == data.shape[0])
    if errors:
        data = data[~data['glycan'].isin(errors)].reset_index(drop=True)
    
    tsne_embeds = TSNE(n_components=2, random_state=seed).fit_transform(embed)
    df_tsne = pd.DataFrame(tsne_embeds, columns=['x', 'y'])  
    df_tsne['glycan'] = data['glycan'].tolist()

    # Select the most relevant categories to see the clusters
    df_tsne['hue'] = data[hue].tolist()
    df_tsne = df_tsne.explode('hue').drop_duplicates(subset=['glycan', 'hue']).reset_index(drop=True)
    top_hues = df_tsne['hue'].value_counts().nlargest(limit).index.tolist()
    df_tsne = df_tsne[df_tsne['hue'].isin(top_hues)].reset_index(drop=True)

    sns.set_theme(rc = {'figure.figsize':(10, 10)}, font_scale=2)
    fig = sns.scatterplot(data=df_tsne, x='x', y='y', hue=hue, palette='colorblind', s=40, rasterized=True)
    fig.set_title('TSNE of Glycan Embeddings')

    return tsne_embeds



    
    
