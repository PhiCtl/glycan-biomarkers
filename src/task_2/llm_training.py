import argparse
from typing import List
from sklearn.model_selection import train_test_split
import wandb
import os, sys
import torch

import pandas as pd

from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, set_seed,\
      RobertaConfig, RobertaForMaskedLM, Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from src.task_2.helpers import load_file
from src.task_2.tokenizer import HuggingFaceTokenizerWrapper
from utils.config import load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GlycanDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling glycan data.
    This dataset tokenizes the input data using a provided tokenizer and prepares it
    for use in training or evaluation tasks. The tokenized data is padded and truncated
    to a specified maximum length.

    Attributes:
        encodings (List[Dict]): A list of tokenized and preprocessed data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves a single sample from the dataset at the specified index.

    Args:
        data (List): A list of input data to be tokenized.
        tokenizer: A tokenizer object used to tokenize the input data.
        max_length (int, optional): The maximum length for tokenized sequences. Defaults to 512.
    """

    def __init__(self, data: List[str], tokenizer, max_length=512):
        super().__init__()
        self.encodings = tokenizer(
            data,
            truncation=True,
            padding='max_length',
            max_length=max_length)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


def prepare_datasets(data: pd.DataFrame, tokenizer, config, seed):
    """
    Prepares training and validation datasets for language model training.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing the input data. 
                             Must include a column named 'glycan'.
        tokenizer: The tokenizer to be used for processing the text data.
        config (dict): A configuration dictionary containing training parameters.
                       Expected keys:
                       - 'training': A dictionary with keys:
                         - 'test_size' (float): Proportion of the dataset to include in the test split.
                         - 'mlm_probability' (float): Probability for masking tokens during 
                           masked language modeling.
        seed (int): Random seed for reproducibility during dataset splitting.
    Returns:
        tuple: A tuple containing:
            - x_tr (GlycanDataset): The training dataset.
            - x_val (GlycanDataset): The validation dataset.
            - data_collator (DataCollatorForLanguageModeling): A data collator for 
              masked language modeling.
    """

    

    assert('glycan' in data.columns)
    df_tr, df_test = train_test_split(data['glycan'].tolist(),\
                                      test_size=config['training']['test_size'],\
                                      random_state=seed)
    x_tr = GlycanDataset(df_tr, tokenizer=tokenizer)
    x_val = GlycanDataset(df_test, tokenizer=tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,\
                                                    mlm=True,\
                                                    mlm_probability=config['training']['mlm_probability'])
    
    return x_tr, x_val, data_collator



def load_training_args(config):
    """
    Loads and initializes the training arguments for a machine learning model.
    Args:
        config (dict): A dictionary containing configuration parameters for training.
            Expected keys in the `config` dictionary:
                - 'training': A nested dictionary with the following keys:
                    - 'output_dir' (str): Directory to save model checkpoints and outputs.
                    - 'logging_dir' (str): Directory to save training logs.
                    - 'num_train_epochs' (int): Number of training epochs.
                    - 'learning_rate' (float): Learning rate for the optimizer.
                    - 'weight_decay' (float): Weight decay for the optimizer.
                    - 'adam_epsilon' (float): Epsilon value for the Adam optimizer.
                    - 'save_steps' (int): Number of steps between saving checkpoints.
    Returns:
        TrainingArguments: An instance of the `TrainingArguments` class initialized with the provided configuration.
    Notes:
        - The function enables training and evaluation by default (`do_train=True`, `do_eval=True`).
        - The output directory is overwritten if it already exists (`overwrite_output_dir=True`).
        - Logging is configured to report to Weights & Biases (`report_to="wandb"`) with a logging step interval of 1.
        - The training run is named 'small-roberta' (`run_name='small-roberta'`).
    """

    # TODO monitor batch size as it might cause unstable training

    training_args = TrainingArguments(

    output_dir=config['training']['output_dir'],
    logging_dir=config['training']['logging_dir'],
    
    num_train_epochs=config['training']['num_train_epochs'],
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    #adam_epsilon=config['training']['adam_epsilon'],
    save_steps=config['training']['save_steps'],

    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    do_train=True,
    do_eval=True,

    report_to="wandb",
    logging_steps = 1,
    run_name = 'small-roberta',
    )

    return training_args

def load_model_config(config):
    """
    Loads and initializes a RobertaConfig object using the provided configuration and tokenizer.
    Args:
        config (dict): A dictionary containing model configuration parameters. 
            Expected keys under 'model' include:
                - 'max_position_embeddings' (int): Maximum number of position embeddings.
                - 'num_attention_heads' (int): Number of attention heads in the model.
                - 'num_hidden_layers' (int): Number of hidden layers in the model.
                - 'type_vocab_size' (int): Size of the type vocabulary.
                - 'is_decoder' (bool): Whether the model is a decoder.
        tokenizer: A tokenizer object with a `vocab_size` attribute.
    Returns:
        RobertaConfig: An instance of the RobertaConfig class initialized with the provided parameters.
    """


    model_config = RobertaConfig(
        vocab_size=config['tokenizer']['vocab_size'],
        max_position_embeddings=config['model']['max_position_embeddings'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        type_vocab_size=config['model']['type_vocab_size'],
        is_decoder=config['model']['is_decoder'],
    )
    return model_config





def main():

    # Get file path
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, required=False, help="data file path",
                        default='data/glycan_embedding/df_glycan.pkl')
    args = parser.parse_args()

    # Initialize training session
    wandb.login()
    os.environ["WANDB_PROJECT"] = "glycan-embedding"
    run = wandb.init( project = "glycan-embedding" )

    # Load config and data
    print("Load config")
    config = load_config()
    seed = config['seed']
    set_seed(seed)
    print("Load data")
    data = load_file(args.file_path)
    config = config['models']['roberta']

    # Load tokenizer
    print("Load tokenizer")
    wrapper = HuggingFaceTokenizerWrapper()
    wrapper.load(config['tokenizer']['path'])
    tokenizer = wrapper.get_tokenizer()


    # Prepare dataset
    print("Prepare datasets")
    x_tr, x_val, data_collator = prepare_datasets(data, tokenizer, config, seed)

    # Prepare training
    print("Prepare training")
    training_args = load_training_args(config)
    model_config = load_model_config(config)
    model = RobertaForMaskedLM(model_config)
    print(f"{model.num_parameters()} model parameters.")
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=x_tr,
    eval_dataset=x_val,
    )

    print(model.config.vocab_size, tokenizer.vocab_size)
    

    # Train
    print("Train model")
    trainer.train()

    # Save model
    print("Save model")
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    trainer.save_model(config['training']['output_dir'])
    run.finish()

if __name__ == "__main__":

    main()