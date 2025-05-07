import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

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

def plot_embeddings(embed:np.ndarray, data:pd.DataFrame, hue:str, limit:int = 5, errors=None, seed=42):

    

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