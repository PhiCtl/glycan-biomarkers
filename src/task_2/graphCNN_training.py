import torch
import time
import copy
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt

from glycowork.ml.models import SweetNet, prep_model
from glycowork.ml.processing import split_data_to_train
from glycowork.ml.train_test_split import hierarchy_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.config import load_config, set_seed
from src.task_2.helpers import load_file, EarlyStopping

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    

def prepare_data(path, class_):
    """
    Prepares data for training and validation by loading, filtering, and splitting it into dataloaders.
    Args:
        path (str): The file path to the pickled DataFrame containing the data.
        class_ (str): The column name in the DataFrame representing the class hierarchy to be used.
    Returns:
        tuple: A tuple containing:
            - dataloaders (dict): A dictionary containing the training and validation dataloaders.
            - class_list (list): A list of unique classes in the specified hierarchy.
            - class_converter (dict): A mapping of class names to their corresponding indices.
    Raises:
        AssertionError: If the specified class_ is not a column in the DataFrame.
    Notes:
        - The function assumes the DataFrame contains hierarchical class information.
        - The `hierarchy_filter` function is used to filter the data based on the specified class hierarchy.
        - The `split_data_to_train` function is used to split the data into training and validation sets.
    """


    df_data = load_file(path, class_)
    train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(df_data,
                                                                              rank = class_, min_seq=10)
    dataloaders = split_data_to_train(train_x, val_x, train_y, val_y)

    return dataloaders, class_list, class_converter


def train_graphCNN(dataloaders, class_type, num_classes, config):
    """
    Trains a Graph Convolutional Neural Network (GraphCNN) model using the provided data loaders and configuration.
    Args:
        dataloaders (dict): A dictionary containing the training and validation data loaders.
        class_type (str): A string representing the type of classification task (e.g., binary or multi-class).
        num_classes (int): The number of output classes for the classification task.
        config (dict): A dictionary containing the configuration parameters for the model, training, and results.
            - config['model']['hidden_dim'] (int): The hidden dimension size for the model.
            - config['training']['learning_rate'] (float): The learning rate for the optimizer.
            - config['training']['weight_decay'] (float): The weight decay (L2 regularization) for the optimizer.
            - config['training']['T_max_scheduler'] (int): The maximum number of iterations for the cosine annealing scheduler.
            - config['training']['patience'] (int): The number of epochs to wait for improvement before early stopping.
            - config['training']['num_train_epochs'] (int): The total number of training epochs.
            - config['results']['models_dir'] (str): The directory path to save the trained model.
    Returns:
        None: The trained model is saved to the specified directory in the configuration.
    Notes:
        - The model architecture used is 'SweetNet'.
        - Early stopping is applied to prevent overfitting.
        - The trained model is saved with a filename that includes the class type.
    """

    save_path = config['training']['save_dir'] 
    os.makedirs(save_path, exist_ok=True)

    model = prep_model('SweetNet',\
                    trained=False, \
                    num_classes=num_classes, \
                    hidden_dim=config['model']['hidden_dim'])
    optimizer = torch.optim.Adam(model.parameters(),\
                                lr = config['training']['learning_rate'],\
                                weight_decay = config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['training']['T_max_scheduler'])
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    early_stopping = EarlyStopping(patience=config['training']['patience'], verbose=True,\
                                path=save_path + f'/Sweetnet_{class_type}.pt')

    model_ft, val_acc, val_losses, epoch  = _train_model(model, dataloaders, criterion, optimizer, scheduler, early_stopping,
                                                        num_epochs = config['training']['num_train_epochs'])
    
    torch.save(model_ft, save_path + f'/Sweetnet_{class_type}.pt')
    plot_metrics(val_acc, epoch, val_losses, save_path)

def plot_metrics(val_acc, epoch, val_losses, save_path=None):

    ## plot loss & accuracy score over the course of training 
    fig, ax = plt.subplots(nrows = 2, ncols = 1) 
    plt.subplot(2, 1, 1)
    plt.plot(range(epoch+1), val_losses)
    plt.title('Training of SweetNet')
    plt.ylabel('Validation Loss')
    plt.legend(['Validation Loss'],loc = 'best')

    plt.subplot(2, 1, 2)
    plt.plot(range(epoch+1), val_acc)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.legend(['Validation Accuracy'], loc = 'best')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/metrics.png")


def _train_model(model, dataloaders, criterion, optimizer, scheduler, early_stopping, num_epochs = 25):
    """
    Trains a PyTorch model using the provided data loaders, criterion, optimizer, and scheduler.
    Args:
        model (torch.nn.Module): The model to be trained.
        dataloaders (dict): A dictionary containing 'train' and 'val' DataLoader objects.
        criterion (torch.nn.Module): The loss function used to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler to adjust the learning rate.
        early_stopping (object): An early stopping object to monitor validation loss and stop training early if needed.
        num_epochs (int, optional): The maximum number of epochs to train the model. Defaults to 25.
    Returns:
        torch.nn.Module: The trained model with the best weights based on validation loss.
    Notes:
        - Tracks and prints the loss, accuracy, and Matthews correlation coefficient (MCC) for both training and validation phases.
        - Saves the model weights with the best validation loss.
        - Implements early stopping based on validation loss.
        - Plots validation loss and accuracy over the course of training.
    Example:
        model = train_model(
            model=my_model,
            dataloaders={'train': train_loader, 'val': val_loader},
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(my_model.parameters()),
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
            early_stopping=early_stopping_instance,
            num_epochs=50
        )
    """


    since = time.time()
    best_loss = 100.00
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    val_losses = []
    val_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = []
            running_acc = []
            running_mcc = []

            for data in dataloaders[phase]:
                x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                edge_index = edge_index.to(DEVICE)
                batch = batch.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(x, edge_index, batch)
                    loss = criterion(pred, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss.append(loss.item())
                pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1)
                running_acc.append(accuracy_score(
                                        y.cpu().detach().numpy().astype(int), pred2))
                running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2))
        
            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_acc)
            epoch_mcc = np.mean(running_mcc)
            print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_mcc))
        
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
                early_stopping(epoch_loss, model)

            scheduler.step()
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))
    model.load_state_dict(best_model_wts)

    return model, val_acc, val_losses, epoch

def main():

    #os.chdir('../..')
    print("Load config..")
    config = load_config()
    set_seed(config)
    
    print("Load data...")
    data_path = os.path.join(config['data']['data_dir_2'], 'df_glycan.pkl')
    dataloaders, class_list, _ = prepare_data(data_path, class_='Family')

    print("Train model")
    train_graphCNN(dataloaders, 'Family', num_classes=len(class_list), config=config['models']['sweetnet'])

if __name__ == '__main__':
    main()