import os
import argparse
import sys
from tqdm import tqdm

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.task_2.loading_helpers import load_file
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

def get_training_corpus(dataset, path, chunk_size=10):
    os.makedirs(path, exist_ok=True)
    
    if len(os.listdir(path)) <= 1:
        logger.info("Preparing files")
        for i in tqdm(range(0, len(dataset), chunk_size)):

            samples = dataset[i: i + chunk_size]
            p = path + f'/glycan_embedding_{i}.txt'

            with open(p, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(sample + '\n')
    logger.info("Done !")
    return [os.path.join(path, f) for f in os.listdir(path)]

class HuggingFaceTokenizerWrapper:
    def __init__(self, tokenizer_type="bpe", max_length=512):
        """Initialize the tokenizer wrapper."""
        self.tokenizer = self._initialize_tokenizer()
        self.max_length = max_length


    def _initialize_tokenizer(self):
        """Initialize tokenizer based on the type."""
        tokenizer = ByteLevelBPETokenizer()
        return tokenizer

    def train(self, files, vocab_size, min_frequency=2, special_tokens=None):
        """Train the tokenizer on given files."""
        if special_tokens is None:
            special_tokens = [ "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
            "[MASK]"
            ]

        self.tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

        logger.info(f"Tokenizer trained with {len(self.tokenizer.get_vocab())} tokens.")

    def save(self, path):
        """Save the trained tokenizer to a specified path."""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_model(path)
        logger.info(f"Tokenizer saved to {path}")
    
    def load(self, path):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(path, max_len=self.max_length)
    
    def get_tokenizer(self):
        return self.tokenizer

        

# Example usage:
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, required=False, help="data file path",
                        default='data/glycan_embedding/df_glycan.pkl')
    args = parser.parse_args()

    # Load data
    logger.info("Load config")
    config = load_config()['models']['roberta']['tokenizer']
    logger.info("Load data")
    data = load_file(args.file_path)['glycan'].values
    paths = get_training_corpus(data, config['files'], chunk_size=1)

    logger.info("Train tokenizer")
    # Initialize wrapper
    tokenizer_wrapper = HuggingFaceTokenizerWrapper(tokenizer_type="bpe")
    
    # Train the tokenizer
    tokenizer_wrapper.train(files=paths, vocab_size=config['vocab_size'])
    
    # Save the tokenizer
    #tokenizer_wrapper.save(config['path'])