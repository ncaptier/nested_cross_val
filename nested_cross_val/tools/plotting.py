import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.axes
from sklearn import metrics
from ._matthews import mcc_f1_curve , mcc_f1_score

def plot_hyperparameters(ncv , name , discrete = True , ax = None , **kwargs):
    """ Plot the number of times each value of an hyperparameter has been selected during the inner tunning process.
    
    Parameters
    ----------
    
    ncv : NestedCV estimator
    
    name : str
        Name of the hyperparameter. It should be the same as in ncv.best_params_
        
    discrete : bool, optional
        The default is True.
        
    ax : matplotlib.axes.Axes object, optional
        The default is None.
    
    **kwargs
        Additional keyword arguments for DataFrame.plot()
        
    Returns
    -------
    None.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10 , 6))
    else : 
        if not isinstance(ax, matplotlib.axes.Axes) :
            warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
            fig, ax = plt.subplots(figsize = (10 , 6))
            
    df = pd.DataFrame(ncv.best_params_)
    
    if discrete :
        if pd.api.types.is_numeric_dtype(df[name]):
            df[name].value_counts().sort_index().plot.bar(rot = 0 , ax = ax , **kwargs)
        else :
            df[name].value_counts().sort_values(ascending = False).plot.bar(rot = 0 , ax = ax , **kwargs)
    else :
        df[name].hist(grid = False , ax = ax , **kwargs)
        
    ax.set_title("Frequency of selections for " + name + " values" , fontsize = 14)
    
    return

def plot_score(ncv , score_name , display , ax = None , **kwargs):
    """ Plot the scores obtained in the outer loop of a nested cross-validation pipeline.
    
    Parameters
    ----------
    ncv : NestedCV estimator
        
    score_name : str
        
    display_method : str {'barplot' , 'boxplot' , 'distribution'}
        
    ax : matplotlib.axes, optional
        The default is None.
    
    **kwargs
        Additional keyword arguments for matplotlib.pyplot.bar(), matplotlib.pyplot.boxplot()
        or matplotlib.pyplot.hist()
        
    Returns
    -------
    None.

    """
    X1 = ncv.outer_results_['training_' + score_name]
    X2 = ncv.outer_results_['test_' + score_name]
        
    if ax is None :
        if display == 'distribution':
            fig, ax = plt.subplots(1 , 3 , figsize = (30 , 6) , sharey = True)
        else :
            fig, ax = plt.subplots(figsize = (10 , 6))
    else :
        if display == 'distribution':
            try :
                ax = ax.flatten()
            except AttributeError :
                warnings.warn("ax should be a numpy array containing at least three matplotlib.axes.Axes objects. It was redefined by default.")
                fig, ax = plt.subplots(1 , 3 , figsize = (30 , 6) , sharey = True)
            else :
                if len(ax) < 3:
                    warnings.warn("ax is not of the right shape. It should contain at least three matplotlib.axes.Axes objects. It was redefined by default.")
                    fig, ax = plt.subplots(1 , 3 , figsize = (30 , 6) , sharey = True)
        else : 
            if not isinstance(ax, matplotlib.axes.Axes) :
                warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
                fig, ax = plt.subplots(figsize = (10 , 6))
    
    if display == 'barplot':
        barWidth = 0.3
        r1 = range(len(X1))
        r2 = [x + barWidth for x in r1]
        ax.bar(r1, X1, width = barWidth, color = ['blue' for i in r1] , label = 'train' , **kwargs)
        ax.bar(r2, X2 , width = barWidth, color = ['red' for i in r1] , label = 'test' , **kwargs)
        ax.set_xticks([r + barWidth / 2 for r in r1])
        ax.set_xticklabels(['out fold ' + str(i) for i in r1]) 
        ax.legend()
        ax.set_title(score_name , fontsize = 14)
        
    elif display =='boxplot':
        boxplotElements = ax.boxplot([X1 , X2] , positions = [1, 3], widths = [1 , 1] 
                                     , patch_artist = True , labels = ['train' , 'test'] , **kwargs)
        colors = ['blue' ,'red']
        for i in range(2): 
            boxplotElements['boxes'][i].set_facecolor(colors[i]) 
            
        ax.set_title(score_name , fontsize = 14)
        
    elif display == 'distribution' :
        ax[0].hist(X1 , bins = 'auto' , color = 'blue' , **kwargs)
        ax[0].set_title(score_name + ' distribution (training)' , fontsize = 14)
        
        ax[1].hist(X2 , bins = 'auto' , color = 'red' , **kwargs)
        ax[1].set_title(score_name + ' distribution (test)' , fontsize = 14)
        
        ax[2].hist(np.array(X1) - np.array(X2) , bins = 'auto' , color = 'green' , **kwargs)
        ax[2].set_title(score_name + ' distribution (training - test)' , fontsize = 14)        
    return

    
def plot_roc(ncv , method = "predict_proba" ,  ax = None):
    """ Plot roc curves
    
    Only available for classifiers
    
    Parameters
    ----------
    ncv : NestedCV estimator
    
    method : str {'predict_proba' , 'predict_log_proba' , 'decision_function'}, optional
            Method to obtain the confidence levels. The default is 'predict_proba'.
        
    ax : matplotlib.axes, optional
        The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10 , 6))
    else : 
        if not isinstance(ax, matplotlib.axes.Axes) :
            warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
            fig, ax = plt.subplots(figsize = (10 , 6))
        
    i = 0
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for scores , y_test in ncv.get_probas(method = method):  
        fpr, tpr, _ = metrics.roc_curve(y_test , scores) 
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        try :
            auc = ncv.outer_results_['test_roc_auc'][i]
        except KeyError :
            auc = metrics.roc_auc_score(y_test , scores) 
        aucs.append(auc)
        ax.plot(fpr , tpr , label = 'out fold ' +str(i) +'  AUC = ' + str(np.round(auc , 3) ))
        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ax.plot(mean_fpr, mean_tpr, color='b', lw = 2 , label = 'Mean ROC mean_AUC = ' + str(np.round(np.mean(aucs) , 3)))
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    
    ax.legend()
    ax.set_xlabel("False positive rate" , fontsize = 12)
    ax.set_ylabel("True positive rate" , fontsize = 12)
    ax.set_title("ROC curve" , fontsize = 14)   
     
    return
    
def plot_PR(ncv , method = "predict_proba" , ax = None):
    """ Plot precision-recall curves
    
    Only available for classifiers
    
    Parameters
    ----------
    ncv : NestedCV estimator
    
    method : str {'predict_proba' , 'predict_log_proba' , 'decision_function'}, optional
            Method to obtain the confidence levels. The default is 'predict_proba'.
            
    ax : matplotlib.axes, optional
        The default is None.

    Returns
    -------
    None.

    """  
    if ax is None:
        fig, ax = plt.subplots(figsize = (10 , 6))
    else : 
        if not isinstance(ax, matplotlib.axes.Axes) :
            warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
            fig, ax = plt.subplots(figsize = (10 , 6))
        
    i = 0
    aps = []
    for scores , y_test in ncv.get_probas(method = method): 
        prec, recall, _ = metrics.precision_recall_curve(y_test , scores) 
        try :
            ap = ncv.outer_results_['test_average_precision'][i]
        except KeyError :
            ap = metrics.average_precision_score(y_test , scores) 
        aps.append(ap)
        ax.plot(recall , prec , label = 'out fold ' +str(i) +'  AP = ' + str(np.round(ap , 3) ))
        i += 1
        
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve" , fontsize = 14)
    return

def plot_mccF1(ncv , method = 'predict_proba' , ax = None) :
    """ Plot Matthews correlation coefficient-F1 score curves
    
    Only available for classifiers
    
    Parameters
    ----------
    ncv : NestedCV estimator
    
    method : str {'predict_proba' , 'predict_log_proba' , 'decision_function'}, optional
            Method to obtain the confidence levels. The default is 'predict_proba'.
            
    ax : matplotlib.axes, optional
        The default is None.

    Returns
    -------
    None.
    
    Note
    -----    
    Fore more details please refer to :
        - https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7 
        - https://arxiv.org/ftp/arxiv/papers/2006/2006.11278.pdf

    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10 , 6))
    else : 
        if not isinstance(ax, matplotlib.axes.Axes) :
            warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
            fig, ax = plt.subplots(figsize = (10 , 6))
        
    i = 0
    for scores , y_test in ncv.get_probas(method = method):
        f1 , mcc , _ = mcc_f1_curve(y_test , scores) 
        try :
            score = ncv.outer_results_['test_mcc_f1'][i]
        except KeyError :
            score = mcc_f1_score(y_test , scores)[0]
        ax.plot(f1 , mcc , label = 'out fold ' +str(i) +'  score = ' + str(np.round(score , 3) ))
        i += 1
        
    ax.legend()
    ax.set_xlabel("F1 score" , fontsize = 12)
    ax.set_ylabel("Matthews score" , fontsize = 12)
    ax.set_title("MCC_F1 curve" , fontsize = 14)
    return


def plot_feature_importances(df , normalize = True , display = 'all' , imp_thr = None , stab_thr = None , handle_negatives = False , ax = None):
    """ Plot feature importances.
    
    Parameters
    ----------
    df : Dataframe, shape (n_features , n_outer_folds)
        Feature importances for each outer fold of a NestedCV object.
        
    normalize : bool, optional
        If True, the feature importances are normalized for each fold. The default is True.
        
    display : str {"all" , "most_important" , "most_stable"}, optional
        Choose the features that will be displayed. The default is 'all'.
        
        - "all" : all features will be displayed.
        - "most_important" : only the most important features will be displayed (the importance is defined with the threshold parameter imp_thr).
          It does not take count of stability, even if a feature is considered important for only one fold, it will be displayed !
        - "most_stable" : among the most important features only those that are considered important for a "significant number of folds"
          (defined by the threshold parameter stab_thr) will be displayed.
        
    imp_thr : Float or int, optional
        Threshold for feature importances. The default is None.
        
        - If imp_thr is a float, all the values above this threshold (or the absolute values if handle_negatives is True) will be selected
        - If imp_thr is an int, for each fold the i highest values (or the absolute values if handle_negatives is True) will be selected 
        
    stab_thr : Float between 0 and 1, optional
        Minimum frequency of importance to display the feature. The default is None.
        
        ex : If stab_thr = 0.5, it means that only the features that are considered important for more than half of the outer folds 
             will be displayed.
    
    handle_negatives : bool, optional
        If True, negatives values are considered important. The default is False
        
        ex : for a logistic regression model, negative coefficients are important for the prediction task.
        
    ax : matplotlib.axes, optional
        The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10 , 6))
    else : 
        if not isinstance(ax, matplotlib.axes.Axes) :
            warnings.warn("ax should be a matplotlib.axes.Axes object. It was redefined by default.")
            fig, ax = plt.subplots(figsize = (10 , 6))
            
    if normalize:
        df = df.apply(lambda x: x/x.sum() , axis = 0)
    
    if display == 'all' :
        
        df.plot.bar(stacked=True , ax = ax)
        ax.set_title("All features" , fontsize = 14)
        
    else:
        
        def _select_features(col , thresold , handle_negatives):
            if handle_negatives : 
                if isinstance(thresold, int):
                    thresold = np.abs(col).sort_values(ascending = False).iloc[thresold]
                return col.where(np.abs(col) > thresold , 0)
            else :
                if isinstance(thresold, int):
                    thresold = col.sort_values(ascending = False).iloc[thresold]
                return col.where(col > thresold , 0)
        
        df = df.apply(_select_features , thresold = imp_thr , handle_negatives = handle_negatives , axis = 0)
        
        if display == 'most_important':
            df[df.sum(axis = 1) >0].plot.bar(stacked = True , ax = ax , rot = 45)
            ax.set_title("Most important features of each fold" , fontsize = 14)
        elif display == 'most_stable': 
            df[((df > 0).sum(axis = 1)/df.shape[1]) >= stab_thr].plot.bar(stacked = True , ax = ax , rot = 45)
            ax.set_title("Most stable important features" , fontsize = 14)
            
        if df.shape[1] > 10 :
            ax.legend().set_visible(False)
    return

