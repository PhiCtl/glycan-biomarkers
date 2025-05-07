import argparse
import logging
import os
import sys

import torch
import wandb
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.task_2.tokenizer import HuggingFaceTokenizerWrapper
from src.task_2.datasets import prepare_datasets
from utils.config import load_config
from utils.logger import get_logger


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(__name__)

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

    training_args = TrainingArguments(

    output_dir=config['training']['output_dir'],
    logging_dir=config['training']['logging_dir'],
    
    num_train_epochs=config['training']['num_train_epochs'],
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    #adam_epsilon=config['training']['adam_epsilon'],
    save_steps=config['training']['save_steps'],

    overwrite_output_dir=True,
    eval_strategy='epoch',
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
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        is_decoder=config['model']['is_decoder'],
        hidden_dropout_prob=config['model']['hidden_dropout_prob'],
        attention_probs_dropout_prob=config['model']['attention_probs_dropout_prob'],
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

    # Load config
    logger.info("Load config")
    config = load_config()
    seed = config['seed']
    set_seed(seed)
    config = config['models']['roberta']

    # Load tokenizer
    logger.info("Load tokenizer")
    wrapper = HuggingFaceTokenizerWrapper()
    wrapper.load(config['tokenizer']['path'])
    tokenizer = wrapper.get_tokenizer()


    # Prepare dataset
    logger.info("Prepare datasets")
    x_tr, x_val, data_collator = prepare_datasets(args.file_path,'RoBERTa',\
                                                tokenizer=tokenizer, config=config, seed=seed)
    logger.info(f"Training size: {len(x_tr)}, validation size: {len(x_val)}")

    # Prepare training
    logger.info("Prepare training")
    training_args = load_training_args(config)
    model_config = load_model_config(config)
    model = RobertaForMaskedLM(model_config)
    logger.info(f"{model.num_parameters()} model parameters.")
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=x_tr,
    eval_dataset=x_val,
    )
    

    # Train
    logger.info("Train model")
    trainer.train()

    # Save model
    logger.info("Save model")
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    trainer.save_model(config['training']['output_dir'])
    run.finish()

if __name__ == "__main__":

    main()