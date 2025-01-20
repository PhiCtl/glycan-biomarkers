import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
import seaborn as sns

import matplotlib.pyplot as plt

def plot_neighbours_distance(df_neighbours, glycans_list,\
                             metric='cosine',\
                             log_scale=False, max_dist=1000, smoothing_sigma=2,\
                             save_fig=False):

    """
    Plots the distances of each glycan to its neighbours and identifies the closest neighbours.

    Args:
        df_neighbours (pd.DataFrame): DataFrame containing the distances between glycans and their neighbours.
        glycans_list (pd.DataFrame): DataFrame containing the list of glycans.
        metric (str): Distance metric to use ('cosine' or 'cityblock'). Default is 'cosine'.
        log_scale (bool): Whether to use logarithmic scale for the x-axis. Default is False.
        max_dist (int): Maximum number of distances to consider. Default is 1000.
        smoothing_sigma (int): Sigma value for Gaussian smoothing. Default is 2.
        save_fig (bool): Whether to save the figure. Default is False.

    Returns:
        dict: Dictionary containing the closest neighbours for each glycan.
    """
    assert('glycan' in glycans_list.columns)
    assert(metric in ['cosine', 'cityblock'])

    neighbours_names = list(set(df_neighbours.columns) - set(glycans_list['glycan'].values))
    assert(0 < max_dist <= len(neighbours_names))

    vecs = df_neighbours[neighbours_names].to_numpy()
    closest_ngb = {g : [] for g in glycans_list['glycan'].values}

    fig, axs = plt.subplots(nrows=glycans_list.shape[0], sharex=True, figsize=(20,10))

    for i, glycan in enumerate(glycans_list['glycan'].values):

        # Compute distances
        glyc_vec = np.expand_dims(df_neighbours[glycan].to_numpy(), axis=1)
        dist = cdist(glyc_vec.T, vecs.T, metric=metric).flatten()
        sorted_idx = np.argsort(dist)
        sorted_dist = np.sort(dist)[:max_dist]

        # Compute potential inflection points
        smoothed_dist = gaussian_filter1d(sorted_dist, sigma=smoothing_sigma)
        dy = np.gradient(smoothed_dist)
        d2y = np.gradient(dy)
        inflection_pts = np.where(np.diff(np.sign(d2y)))[0]

        # Plot
        axs[i].plot(sorted_dist)
        axs[i].scatter(inflection_pts, [sorted_dist[i] for i in inflection_pts])
        # axs[i].set(xlim=(1,max_dist))
        axs[i].set_title(glycan[:15] + '...')
        axs[i].set_ylabel(metric+ ' distance')
        if log_scale : axs[i].set_xscale('log')

        # retrieve neighbours before first, second and th inflection point
        closest_ngb[glycan].append([neighbours_names[i] for i in sorted_idx[:inflection_pts[0]+1]])
        closest_ngb[glycan].append([neighbours_names[i] for i in sorted_idx[:inflection_pts[1]+1]])
        closest_ngb[glycan].append([neighbours_names[i] for i in sorted_idx[:inflection_pts[2]+1]])

    plt.suptitle('Glycans distance to neighbours')
    plt.tight_layout()
    if save_fig : plt.savefig(f"/content/{metric}_distance.jpg")
    plt.show()

    return closest_ngb


# Run DBSCAN with the chosen epsilon value
def run_dbscan(df_data, embed, eps, min_samples, label='type', metric='cosine'):
    """
    Runs DBSCAN and visualizes the clusters.
    Args:
        df_data (pd DataFrame): Data embeddings to cluster with data points name in index.
        embed (pd DataFrame): Data visualisation embeddings and labels with data points name in index.
        eps (float): Epsilon parameter for DBSCAN.
        min_samples (int): Minimum number of samples for a cluster.
    """
    assert((df_data.shape[0] == embed.shape[0])) # check for consistency in number of sample
    assert(len(set(df_data.index) - set(embed.index)) == 0) # check for consistency in index
    assert(label in embed.columns)

    # Cluster the data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(df_data.to_numpy())
    
    # Visualize the clustering
    augm_embed = embed.loc[df_data.index] # make sure that it is ordered the same
    augm_embed['cluster'] = labels

    plt.figure(figsize = (9, 9))
    sns.scatterplot(data=augm_embed, x=0, y=1, hue='cluster', style=label)

    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples}) yields {len(np.unique(labels))} clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()

def glycan_information(glycan, neighbours, df_metadata, metadata, df_proteins_binding, closeness_level=1, print_agg=False):
    """
    Analyzes and prints information about a given glycan and its neighbours, including metadata and protein binding data.
    Parameters:
    glycan (str): The glycan of interest.
    neighbours (dict): A dictionary where keys are glycans and values are lists of neighbouring glycans.
    df_metadata (pd.DataFrame): DataFrame containing metadata about glycans.
    metadata (list): List of metadata columns to be considered.
    df_proteins_binding (pd.DataFrame): DataFrame containing protein binding data for glycans.
    closeness_level (int, optional): The level of closeness to consider for neighbours. Defaults to 1.
    print_agg (bool, optional): If True, prints aggregated metadata for each neighbour. Defaults to False.
    Returns:
    pd.DataFrame: DataFrame containing metadata and protein binding information for the glycan and its neighbours.
    """

    assert(isinstance(glycan, str))
    assert(isinstance(neighbours, dict) & (glycan in neighbours.keys()))
    assert(isinstance(df_metadata, pd.DataFrame))
    assert(isinstance(metadata, list))
    assert(isinstance(df_proteins_binding, pd.DataFrame))

    neighbours_metadata = pd.DataFrame({'neighbours' : [i for j in neighbours[glycan] for i in j],\
                                        'infl_order':[1]*len(neighbours[glycan][0]) +
                                                        [2]*len(neighbours[glycan][1]) +
                                                        [3]*len(neighbours[glycan][2])})
    neighbours_metadata = neighbours_metadata.merge(df_metadata[metadata + ['glycan']].drop_duplicates('glycan'),\
                                                    left_on='neighbours', right_on='glycan', how='left')\
                                            .query('infl_order <= @closeness_level')\
                                            .drop('glycan', axis=1)\
                                            .drop_duplicates('neighbours')
    
    print(f"For glycan {glycan} and its {neighbours_metadata['neighbours'].nunique()} neighbours:\n", "\n")
    if print_agg:
        for c in metadata:
            print("###", c, "\n")
            for _, g in neighbours_metadata.iterrows():
                print(g['neighbours'], ": ")
                print(g[c], "\n")
            print("*"*len(g['neighbours']), "\n")

    else : 
        for c in metadata:
            d = neighbours_metadata[[c]].dropna()
            print(d.explode(c).value_counts(), "\n")
    
    # Find neighbouring glycans for which we have protein binding data
    glycans_binding = list(set(neighbours_metadata['neighbours'].drop_duplicates()).intersection(set(df_proteins_binding.columns)))
    proteins = set()
    bindings = {g : [] for g in glycans_binding}
    for g in glycans_binding:
        p = list(df_proteins_binding[~df_proteins_binding[g].isna()].set_index('protein').index)
        proteins.update(p)
        bindings[g] = p
    print(f"{len(proteins)} known interacting proteins among catalogued glycans are common to all neighbours.")

    # Return metadata
    df_bindings = pd.DataFrame({'neighbours':[g for g in bindings.keys()], 'proteins': [bindings[g] for g in bindings.keys()]})
    neighbours_metadata = neighbours_metadata.merge(df_bindings, on='neighbours', how='outer')
    
    return neighbours_metadata

def plot_embedding_classes(df_embedding, embedding_type, visualization_hue='from_human', save_fig=False):
    """
    Plots the embedding of classes with different markers and colors based on their type.
    Parameters:
    df_embedding (pd.DataFrame): DataFrame containing the embedding data. Must include columns 'type' and the specified visualization_hue.
    embedding_type (str): The type of embedding used, used for labeling the axes.
    visualization_hue (str, optional): Column name in df_embedding to use for coloring the points. Default is 'from_human'.
    save_fig (bool, optional): If True, saves the figure as a .jpg file. Default is False.
    Returns:
    None
    """
  
    assert((visualization_hue in df_embedding.columns) & ('type' in df_embedding.columns))

    plt.figure(figsize = (9, 9))
    sns.scatterplot(data = df_embedding[df_embedding['type'] == 'known'], x =0 , y = 1,\
                s = 40, alpha = 0.8, hue=visualization_hue,\
                marker='+', palette =sns.color_palette("pastel") )
    sns.scatterplot(data = df_embedding[df_embedding['type'] == 'N_glycans'], x =0 , y = 1,\
                    s = 120, alpha = 1,palette = ['red'],\
                    marker='^', label='N_glycans')
    fig = sns.scatterplot(data = df_embedding[df_embedding['type'] == 'unknown'], x =0 , y = 1,
                    s = 80, alpha = 1,\
                    palette = ['dark'], marker='o', label='unknown')
    
    for i, point in df_embedding.iterrows():
        if point['type'] == 'unknown':
            fig.text(point[0]+.03, point[1], str(i)[:15] + '...')

    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xlabel(f'{embedding_type} Dim1')
    plt.ylabel(f'{embedding_type} Dim2')
    plt.title('Visualizing glycans clusters')
    plt.tight_layout()
    if save_fig : plt.savefig(f"/content/glycans_embeddings_clusters_{visualization_hue}.jpg")