import warnings

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering


def collect_importances(ncv, estimator, importances, feature_names=None):
    """ Collect the feature importances through the outer folds of a fitted NestedCV object.

    Parameters
    ----------
    ncv : NestedCV estimator

    importances : str or callable
        Method to obtain the feature importances (see get_attributes() method of NestedCV).

        ex : importances = "coef_" (for linear models)

    feature_names : list of strings, optionnal
        The default is None.

    Returns
    -------
    Dataframe, shape (n_features , n_outer_folds)

    Note
    ------
    ncv.get_attributes(importances) should return an array of shape (n_features, ). In particular, for estimators which integrate a feature
    selection step as pre-processing, features which were discarded should be re-integrated in the feature importances array (ex : using NaN).

    """
    temp = {}
    i = 0
    for imp in ncv.get_attributes(estimator=estimator, attribute=importances):
        temp['out fold ' + str(i)] = pd.Series(imp, index=feature_names)
        i += 1
    return pd.DataFrame(temp)


def collect_meta_importances(ncv, estimator, importances, components, projections=None, full_components=None,
                             full_projections=None
                             , clustering_space="features", consensus="centrotypes", plot=False, plot_method="mds",
                             ax=None):
    """ Collect and cluster the components obtained from the decomposition (ex : PCA) of the trainign data for each outer fold.

    Parameters
    ----------
    ncv : NestedCV estimator

    importances : str or callable
        Method to obtain the feature importances (see get_attributes() method of NestedCV).

        ex : importances = importances = "coef_" (for linear models)

    components : str or callable
        Method to obtain the components (see get_attributes() method of NestedCV).

        ex : components = (lambda pipeline : pipeline['PCA'].components_)

    projections : str or callable, optional
        Method to obtain the projection of the full data_set in the component space (see get_attributes() method of NestedCV).
        It should return an array of shape (n_samples , n_components).
        The default is None.

        ex : projections = (lambda pipeline : pipeline['PCA'].transform(ncv.X_)). The default is None.

    full_components : array, shape (n_components, n_features), optional
        Components coming from the decomposition of the full data set ncv.X_. The default is None.

    full_projections : array, shape (n_samples , n_components), optional
        Projections of the full data set ncv.X_. The default is None.

    clustering_space : str {"features" , "samples"}, optional
        The default is "features".

        - "features" : each component will be represented by a vector (n_features, ) and the distances between these vectors
          will be used to cluster the components.
        - "Samples" : each component will be represented by a vector (n_samples, ) and the distances between these vectors
          will be used to cluster the components.

        If clustering_space = "samples", the value of the projections parameter should not be None !

    consensus : str {"centrotypes" , "full"}, optional
        Strategy to characterize each cluster. The default is "centrotypes".

        - "centrotypes" :  each cluster will be characterized by its centrotype
        - "full" : each cluster will be characterized by the component coming from the decomposition of the full data set which
          belongs to this cluster.

        If consensus = "full", the value of the full_components parameter should not be None. Besides, if clustering_space = "samples", the
        value of the full_projections parameter should also not be None !

    plot : bool, optional
        If True, plot the projection of all the components in 2D. The default is False.

    plot_method : str {"mds" , "tsne" , "umap"}, optional
        See _projection. The default is "mds".

    ax : matplotlib.axes, optional
        The default is None.

    Returns
    -------
    df : Dataframe, shape (n_clusters , n_outer_folds)
        Feature importances for each outer fold of a NestedCV object.

    stabilities : Series, shape (n_clusters)
        Stability indexes of the clusters.

    Consensus : Dataframe, shape (n_clusters , n_features)
        Consensus components that characterize/represent the different clusters.

    Note
    ------

    Please note that this function is quite rigid and difficult to manipulate. We will try to improve it in the future !

    """
    # Collect components/meta-features accross the folds
    L = [x for x in ncv.get_attributes(estimator=estimator, attribute=components)]
    if full_components is not None:
        L.append(full_components)
    C = np.vstack(L)
    n_components = L[0].shape[0]

    # Compute similarities between meta-features (either using the feature space or the "sample" space)
    if clustering_space == "features":
        Sim = np.abs(np.corrcoef(x=C, rowvar=True))
    elif clustering_space == "samples":
        L = [x.T for x in ncv.get_attributes(estimator=estimator, attribute=projections)]
        if full_projections is not None:
            L.append(full_projections.T)
        Sim = np.abs(np.corrcoef(x=np.vstack(L), rowvar=True))

    clustering = AgglomerativeClustering(n_clusters=n_components, affinity="precomputed", linkage='average').fit(
        1 - Sim)

    # Associate each cluster with its feature importance for each fold
    temp = {}
    i = 0
    for imp in ncv.get_attributes(estimator=estimator, attribute=importances):
        index = ['cluster ' + str(j) for j in clustering.labels_[i * n_components: (i + 1) * n_components]]
        s = pd.Series(imp, index=index)
        temp['out fold ' + str(i)] = s.groupby(s.index).mean()
        i += 1
    df = pd.DataFrame(temp)

    # Compute the stability index of each cluster
    stabilities = np.zeros(n_components)
    for i in range(n_components):
        cluster_labels = list(np.argwhere(clustering.labels_ == i).flatten())
        stabilities[i] = _stability_index(Sim, cluster_labels)
    stabilities = pd.Series(stabilities, index=['cluster ' + str(i) for i in range(n_components)])

    # Compute a consensus meta-feature for each cluster
    if consensus == "centrotypes":

        Consensus = np.zeros((n_components, C.shape[1]))
        for i in range(n_components):
            cluster_labels = list(np.argwhere(clustering.labels_ == i).flatten())
            Consensus[i, :] = _centrotype(C, Sim, cluster_labels)
        Consensus = pd.DataFrame(Consensus, index=['cluster ' + str(i) for i in range(n_components)])

    elif consensus == "full":

        Consensus = pd.DataFrame(full_components,
                                 index=['cluster ' + str(j) for j in clustering.labels_[- full_components.shape[0]:]])

        # Plot a 2D projection of all the components
    if plot:
        if (full_components is not None) or (full_projections is not None):
            _projection(Sim, clustering.labels_, method=plot_method
                        , specific_idx=np.arange(Sim.shape[0], dtype=int)[- full_components.shape[0]:], ax=ax)
        else:
            _projection(Sim, clustering.labels_, method=plot_method, ax=ax)

    return df, stabilities, Consensus


def _projection(Sim, cluster_labels, method="mds", specific_idx=None, ax=None):
    """ Project all the components in 2D using their similarity matrix.

    Parameters
    ----------
    Sim : 2D array, shape (n_components , n_components)
        Similarity matrix for the components

    cluster_labels : ndarray of shape (n_components)
        cluster labels for each point

    method : string, optional
        Name of the dimensionality reduction method (e.g "tsne" , "mds" or "umap")
        The default is "umap".

    specific_idx : array-like, optional
        Define specific components that will be represented with a red star on the 2D projection plot.
        The default is None.

    ax : matplotlib.axes, optional
        The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    elif not isinstance(ax, matplotlib.axes.Axes):
        warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
        fig, ax = plt.subplots(figsize=(10, 6))

    if method == "tsne":
        embedding = manifold.TSNE(n_components=2, metric="precomputed")
    elif method == "mds":
        embedding = manifold.MDS(n_components=2, dissimilarity="precomputed", n_jobs=-1)
    elif method == "umap":
        embedding = umap.UMAP(n_components=2, metric="precomputed")

    P = embedding.fit_transform(np.sqrt(1 - Sim))

    if specific_idx is not None:
        other_idx = list(set(range(P.shape[0])) - set(specific_idx))
        ax.scatter(P[other_idx, 0], P[other_idx, 1], c=cluster_labels[other_idx], cmap='viridis',
                   label="ncv components")
        ax.scatter(P[specific_idx, 0], P[specific_idx, 1], marker="*", s=100, c="red", label="full components")
        ax.legend()
    else:
        ax.scatter(P[:, 0], P[:, 1], c=cluster_labels, cmap='viridis')
    ax.set_title("2D embedding of the clusters of components (" + method + ")", fontsize=14)
    return


def _stability_index(Sim, cluster_labels):
    """Compute the stability index for the cluster of components defined by cluster_labels.

    Parameters
    ----------
    Sim : 2D array, shape (n_components , n_components)
        Similarity matrix

    cluster_labels : list of integers
        Indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    Float
        Stability index for the cluster of components defined by cluster_labels

    """
    temp = Sim[np.ix_(cluster_labels, cluster_labels)]
    ex_cluster = list(set(range(Sim.shape[1])) - set(cluster_labels))

    # aics = average intra-cluster similarities
    aics = (1 / len(cluster_labels) ** 2) * np.sum(temp)

    # aecs = average extra-cluster similarities
    aecs = (1 / (len(ex_cluster) * len(cluster_labels))) * np.sum(Sim[np.ix_(cluster_labels, ex_cluster)])

    return aics - aecs


def _centrotype(X, Sim, cluster_labels):
    """Compute the centrotype of the cluster of components defined by cluster_labels

       centrotype : component of the cluster which is the most similar to the other components
                   of the cluster
    Parameters
    ----------
    X : 2D array, shape (n_components , n_observations)

    Sim : 2D array, shape (n_components , n_components)
        Similarity matrix for the components (i.e rows of X)

    cluster_labels : list of integers
        Indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    1D array, shape (n_observations)
        Centrotype of the cluster of components defined by cluster_labels

    """
    temp = np.argmax(np.sum(Sim[np.ix_(cluster_labels, cluster_labels)], axis=0))
    return X[cluster_labels[temp], :]
