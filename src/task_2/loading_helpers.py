import os
import pandas as pd
from transformers import RobertaForMaskedLM, RobertaConfig
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_file(path):
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
        return loaders[ext](path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_model(path, model_type: str, config: dict):
        
    if model_type == 'SweetNet':
        import torch
        model_ft = torch.load(path, map_location=DEVICE, weights_only=False)
        
    elif model_type == 'RoBERTa':
        model_config = RobertaConfig.from_pretrained(config['training']['output_dir'])
        model_config.output_hidden_states = True
        model_config.vocab_size = config['tokenizer']['vocab_size']
        model_config.max_position_embeddings = config['model']['max_position_embeddings']
        model_ft = RobertaForMaskedLM.from_pretrained(config['training']['output_dir'], config=model_config)
    
    else:
        raise NotImplementedError()

    model_ft.eval()
    return model_ft



    
    
