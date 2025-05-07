import torch 
from torch.utils.data import Dataset
from typing import List
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from src.task_2.loading_helpers import load_file

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
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


def prepare_datasets(path, model_type: str, **kwargs):
    """
    Prepares data for training and validation by loading, filtering, and splitting it into dataloaders or datasets.
    Args:
        path (str): The file path to the pickled DataFrame containing the data.
        class_ (str): The column name in the DataFrame representing the class hierarchy to be used.
        data (pd.DataFrame): A pandas DataFrame containing the input data. Must include a column named 'glycan'.
        tokenizer: The tokenizer to be used for processing the text data.
        config (dict): A configuration dictionary containing training parameters.
                       Expected keys:
                       - 'training': A dictionary with keys:
                         - 'test_size' (float): Proportion of the dataset to include in the test split.
                         - 'mlm_probability' (float): Probability for masking tokens during 
                           masked language modeling.
        seed (int): Random seed for reproducibility during dataset splitting.
        model_type (str): The type of model being used ('SweetNet' or 'RoBERTa').
    Returns:
        tuple: A tuple containing:
            - For 'RoBERTa':
                - x_tr (GlycanDataset): The training dataset.
                - x_val (GlycanDataset): The validation dataset.
                - data_collator (DataCollatorForLanguageModeling): A data collator for masked language modeling.
            - For 'SweetNet':
                - dataloaders (dict): A dictionary containing the training and validation dataloaders.
                - class_list (list): A list of unique classes in the specified hierarchy.
                - class_converter (dict): A mapping of class names to their corresponding indices.
    Raises:
        AssertionError: If the specified class_ is not a column in the DataFrame.
    Notes:
        - The function assumes the DataFrame contains hierarchical class information for 'SweetNet'.
        - The `hierarchy_filter` function is used to filter the data based on the specified class hierarchy for 'SweetNet'.
        - The `split_data_to_train` function is used to split the data into training and validation sets for 'SweetNet'.
    """

    data = load_file(path)

    if model_type == 'RoBERTa':
        
        assert(('tokenizer' in kwargs.keys()) & ('config' in kwargs.keys()) & ('seed' in kwargs.keys())), \
            "Tokenizer, config and seed are required for RoBERTa model."
        tokenizer = kwargs['tokenizer']
        config = kwargs['config']
        seed = kwargs['seed']

        df_tr, df_test = train_test_split(data['glycan'].tolist(),
                                          test_size=config['training']['test_size'],
                                          random_state=seed)
        x_tr = GlycanDataset(df_tr, tokenizer=tokenizer)
        x_val = GlycanDataset(df_test, tokenizer=tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                        mlm=True,
                                                        mlm_probability=config['training']['mlm_probability'])
        return x_tr, x_val, data_collator

    elif model_type == 'SweetNet':

        from glycowork.ml.processing import split_data_to_train
        from glycowork.ml.train_test_split import hierarchy_filter

        assert('class_' in kwargs.keys())
        class_ = kwargs['class_']
        data = data.explode(class_)
        train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(data,
                                                                                              rank=class_,
                                                                                              min_seq=10)
        dataloaders = split_data_to_train(train_x, val_x, train_y, val_y)
        return dataloaders, class_list, class_converter

    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")