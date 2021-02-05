import numpy as np
import warnings
import time
from shutil import rmtree
from sklearn.base import clone , check_array , BaseEstimator , MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn import metrics
from joblib import Parallel, delayed, Memory

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.model_selection import RandomizedSearchCV as dask_RandomizedSearchCV
from sklearn.model_selection import check_cv , GridSearchCV , RandomizedSearchCV


class NestedCV(BaseEstimator , MetaEstimatorMixin):
    """ Implement a nested cross-validation pipeline compatible with sklearn API.
    
    Parameters
    ----------
    
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
          
    params : dict
        Dictionary with parameters names (str) as keys and distributions (if randomized = True) or lists of parameters 
        to try (otherwise). See sklearn GridSearchCV and RandomizedSearchCV for more details.
        
    cv_inner : int, cross-validation generator or an iterable
        Determines the inner cross-validation splitting strategy. Possible inputs for cv are:
            - integer, to specify the number of folds in a (Stratified)KFold
            - CV splitter (see https://scikit-learn.org/stable/glossary.html#term-CV-splitter)
            - An iterable yielding (train, test) splits as arrays of indices
            
    cv_outer : int, cross-validation generator or an iterable
        Determines the outer cross-validation splitting strategy. Possible inputs for cv are:
            - integer, to specify the number of folds in a (Stratified)KFold
            - CV splitter (see https://scikit-learn.org/stable/glossary.html#term-CV-splitter)
            - An iterable yielding (train, test) splits as arrays of indices
            
    scoring_inner : str, callable
        Strategy to evaluate the performance of the model on in the inner loop.
        See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for more details.
    
    scoring_outer : str, callable, dict of strings and callables
        Strategy to evaluate the performances of the model in the outer loop. A dictionnary can be used for
        multiple scores.
        
    n_jobs_inner : int, optionnal
        number of jobs to run in parallel for each inner cross-validation step. -1 means using all processors.
        See the joblib package documentation for more explanations. 
        The default is -1.
        
    n_jobs_outer : int, optionnal
        number of jobs to run in parallel for the outer loop. -1 means using all processors.
        See the joblib package documentation for more explanations. 
        The default is -1.
    
    refit_estimators : bool, optionnal
        If True, refit the best estimator for each outer train-test split (on the whole train data set) 
        The default is True.
    
    randomized : boolean, optionnal
        If True, a randomized search strategy is used. Otherwise, a grid search strategy is used.
    
    library : str {"sklearn" , "dask"}, optionnal
        If library = "sklearn" RandomSearchCV or GridSearchCV classes are imported from sklearn
        else if library = "dask" they are imported from dask_ml.
        The default is "sklearn".

    caching : bool, optionnal
        If True, a caching directory is created for each cross-validation in the inner loop and the 
        operations are saved to avoid unecessary repetitions. 
        See https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation for more details.
        Only available when library = "sklearn" and for Pipeline estimators.
        The default is False.
    
    scheduler : string, callable, Client, or None, optionnal.
       The dask scheduler to use. 
       See https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.html for more details.
       Only available when library = "dask".
       The default is None.
       
    Attributes
    ----------
            
    best_params_ : list
        Params settings that were selected by cross-validation for each outer train-test split.
        
    best_estimators_ : list
        Estimators that were selected by cross-validation. If refit_estimators is True each one is refitted on the full training data
        (i.e with its associated outer train-test split).
        
    inner_results_ : dict
        Nested dictionary that saves all the results for each inner cross-validation (one for each outer train-test split).
        
    outer_results_ : dict
        
    X_ : array-like, shape (n_samples , n_features)
        Save the training data for additional computations once the method is fitted.
        
    y_ : array-like, shape (n_samples , n_output)
        Save the training data for additional computations once the method is fitted.
        
    cv_outer_ : cross-validation generator
        Save the outer cross-validation shceme (with fixed random state) for additional computations once the method is fitted
        
    Notes
    -----
    
    Dask_ML optimizes computational time and memory during hyperparameter optimization
    See https://ml.dask.org/hyper-parameter-search.html  and https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.html
    for more details
    
    """
    def __init__(self, estimator, params , cv_inner , cv_outer , scoring_inner , scoring_outer , n_jobs_inner = -1 , 
                 n_jobs_outer = -1 , refit_estimators = True , randomized = False , library = 'sklearn' , caching = False , scheduler = None):
            
        self.estimator = estimator
        self.params = params
        self.n_jobs_inner = n_jobs_inner
        self.n_jobs_outer = n_jobs_outer
        self.refit_estimators = refit_estimators
        self.randomized = randomized
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer
        self.scoring_inner = scoring_inner
        self.scoring_outer = scoring_outer
        self.library = library
        self.caching = caching
        self.scheduler = scheduler
    
    def _check_is_fitted(self, method_name):
        if not self.refit_estimators:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit_estimators=False. %s is '
                                 'available only if all the estimators in best_estimators_'
                                 'were fitted. You can refit the estimators '
                                 'manually using the ``best_estimators_``, "X_" , "y_" and "cv_outer_" '
                                 'attributes.'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self)
            
    def _inner_dask(self , X , y):
        """ Perform hyperparameter tuning with cross-validation for each outer fold (using dask-ml library).
        Each time, save best parameters and associated best estimator (not refitted !).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data set.
            
        y : array-like, shape (n_samples, n_output)
            Target relative to X for classification or regression.
        """
        
        cv_estimator = clone(self.estimator)
        
        if self.randomized :
            inner = dask_RandomizedSearchCV(estimator = cv_estimator , param_distributions = self.params , scoring = self.scoring_inner
                                , cv = self.cv_inner , n_jobs = self.n_jobs_inner , refit = False , scheduler = self.scheduler)
        else :
            inner = dask_GridSearchCV(estimator = cv_estimator , param_grid = self.params , scoring = self.scoring_inner
                                , cv = self.cv_inner , n_jobs = self.n_jobs_inner , refit = False , scheduler = self.scheduler)   
            
        count = 0        
        for train , _ in self.cv_outer_.split(X , y):                         
            X_train , y_train = X[train , :] , y[train]  
            start = time.time()
            inner.fit(X_train , y_train)    
            minutes, seconds = divmod(time.time() - start, 60)
            print("inner training out fold " + str(count) + " done !" , "running time (min): " + "{:0>2}:{:05.2f}".format(int(minutes),seconds)) 
            self.inner_results_['out fold ' + str(count)] = inner.cv_results_
            
            #best_params_ attribute is not provided with dask-ml method when refit = False, we need to extract it manually
            best_params_ = inner.cv_results_["params"][inner.cv_results_["rank_test_score"].argmin()]
            self.best_params_.append(best_params_)
            self.best_estimators_.append(clone(clone(self.estimator).set_params(**best_params_)))          
            count += 1
        return self
    
    def _inner_sklearn(self , X , y):
        """ Perform hyperparameter tuning with cross-validation for each outer fold (using sklearn library). 
        Each time, save best parameters and associated best estimator (not refitted !).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data set.
            
        y : array-like, shape (n_samples, n_output)
            Target relative to X for classification or regression.
        """
        
        if self.caching :        
            location = 'cachedir'
            memory = Memory(location = location , verbose = 0 , mmap_mode = 'r+')
            try :
                cv_estimator = clone(self.estimator).set_params(memory = memory)
            except ValueError as e:
                print(e)
                warnings.warn("Caching is only available with an estimator built with sklearn.pipeline.Pipeline (see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html for more details). This argument was ignored by default.")
                cv_estimator = clone(self.estimator)
            else :
                if self.n_jobs_inner != 1 :
                    warnings.warn("A combined use of joblib.Memory (function caching) and joblib.Parallel should be considered with care since it may lead to some failures. In particular, we noticed some errors when dealing with custom transformers in the pipeline.")                    
        else :
            cv_estimator = clone(self.estimator)
        
        if self.randomized :
            inner = RandomizedSearchCV(estimator = cv_estimator , param_distributions = self.params , scoring = self.scoring_inner
                                , cv = self.cv_inner , n_jobs = self.n_jobs_inner , refit = False , verbose = 1)
        else :
            inner = GridSearchCV(estimator = cv_estimator , param_grid = self.params , scoring = self.scoring_inner
                                , cv = self.cv_inner , n_jobs = self.n_jobs_inner , refit = False , verbose = 1)   
            
        count = 0        
        for train , _ in self.cv_outer_.split(X , y):                        
            X_train , y_train = X[train , :] , y[train]           
            inner.fit(X_train , y_train)       
            self.inner_results_['out fold ' + str(count)] = inner.cv_results_
            self.best_params_.append(inner.best_params_)
            self.best_estimators_.append(clone(clone(self.estimator).set_params(**inner.best_params_)))
            if self.caching :
                memory.clear(warn=False)
                rmtree(location)
            count += 1
        return self
    
    
    def fit(self, X , y):
        """ Fit the nested cross-validation with X and y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data set.
            
        y : array-like, shape (n_samples, n_output)
            Target relative to X for classification or regression.
        """
        
        # 1. Initialisation
        
        self.X_ = check_array(X , force_all_finite="allow-nan")
        self.y_ = check_array(y, ensure_2d=False, dtype=None)       
        self.cv_outer_ = _validate_cv(self.cv_outer)
        
        self.inner_results_ = {}
        self.outer_results_ = {}
        self.best_params_ = []
        self.best_estimators_ = []
                    
        # 2. Inner loops (tuning by cross-validation)
        
        if self.library == 'sklearn':
            self._inner_sklearn(self.X_ , self.y_)
        elif self.library == 'dask':
            self._inner_dask(self.X_ , self.y_)
            
            
        # 3. Outer loop (cross-validation)

        if self.refit_estimators : 
            parallel = Parallel(n_jobs=self.n_jobs_outer , require ='sharedmem')        
            with parallel:
                outer = parallel(delayed(_fit_out)(estimator, self.X_ , self.y_ , train , test , self.scoring_outer)
                               for (train , test , estimator) in _enumerate(self.cv_outer_.split(self.X_ , self.y_) , self.best_estimators_))
        else :
            parallel = Parallel(n_jobs=self.n_jobs_outer)
            with parallel:
                outer = parallel(delayed(_fit_out)(clone(estimator), self.X_ , self.y_ , train , test , self.scoring_outer)
                               for (train , test , estimator) in _enumerate(self.cv_outer_.split(self.X_ , self.y_) , self.best_estimators_))
        
        # 4. Format results

        L = list(zip(*outer))     
        self.outer_results_.update(_aggregate_score_dicts(L[0], name = 'training'))
        self.outer_results_.update(_aggregate_score_dicts(L[1] , name = 'test'))
        
        return self
    
    
    def add_scores(self, scoring):
        """ Add scores to the outer_results_ dictionary (using the best estimator for each outer train-test split).
        
        Only available is refit_estimators = True.
        
        Parameters
        ----------
        scoring : str, callable, dict of strings and callables
            Strategy to evaluate the performances of the model in the outer loop. 
            A dictionnary can be used for multiple scores.

        Returns
        -------
        None.

        """
        self._check_is_fitted('add_scores')
        
        i = 0
        s = []
        for train , test in self.cv_outer_.split(self.X_ , self.y_):  
            X_train , y_train , X_test , y_test = self.X_[train , :] , self.y_[train] , self.X_[test , :] , self.y_[test]
            estimator = self.best_estimators_[i]
            s.append(_compute_scores(estimator , X_train , y_train , X_test , y_test , scoring))
        L = list(zip(*s))     
        self.outer_results_.update(_aggregate_score_dicts(L[0], name = 'training'))
        self.outer_results_.update(_aggregate_score_dicts(L[1] , name = 'test'))
        return 
    
    def get_predictions(self):
        """ Collect predictions for each estimator in best_estimators_ 
        
        Only available is refit_estimators = True.

        Yields
        -------
        (array of shape (n_samples) or (n_samples , n_outputs) , array of shape (n_samples) or (n_samples , n_outputs))
            Tuple containing y_pred and y for each train-test split defined by cv_outer_.split(X_ , y_)

        """
        self._check_is_fitted('get_predictions')
        i = 0
        for _ , test in self.cv_outer_.split(self.X_ , self.y_):  
            estimator = self.best_estimators_[i]
            yield(estimator.predict(self.X_[test , :]) , self.y_[test])
    
    def get_probas(self , method = 'predict_proba'):
        """ Collect confidence values for each estimator in best_estimators_ 
        
        Only available is refit_estimators = True.
        
        Parameters
        ----------
        method : str {'predict_proba' , 'predict_log_proba' , 'decision_function'}, optional
            Method to obtain the confidence levels. The default is 'predict_proba'.

        Yields
        -------
        (array of shape (n_samples , n_classes) , array of shape (n_samples))
            Tuple containing y_probas and y for each train-test split defined by cv_outer_.split(X_ , y_)
        
        Note
        -----
        Only available for classifiers
        
        """
        
        self._check_is_fitted('get_probas')
        i = 0
        for _ , test in self.cv_outer_.split(self.X_ , self.y_):  
            estimator = self.best_estimators_[i]
            probas = getattr(estimator , method)(self.X_[test , :])         
            if len(probas.shape) == 2 and probas.shape[1] == 2:
                probas = probas[: , 1]                
            yield(probas , self.y_[test])
          
    def get_attributes(self , attribute):
        """ Collect attribute for each estimator in best_estimators_.
        
        Only available is refit_estimators = True.
        
        Parameters
        ----------
        attribute : str or callable
            If attribute is a string, it should correspond to a valid attribute of the estimators 
            contained in self.best_estimators_.

        Yields
        ------
        Any
            object defined by the parameter "attribute".

        """
        self._check_is_fitted('get_attributes')
        
        for estimator in self.best_estimators_ :
            if isinstance(attribute, str):
                yield getattr(estimator , attribute)
            else :
                yield attribute(estimator)




def _fit_out(estimator , X , y , train , test , scoring):
    """ Fit and score an estimator on a train-test split of (X , y)

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        
    X : array-like, shape (n_samples , n_features)

    y :  array-like, shape (n_samples , n_output)

    train : ndarray
        The training set indices for that split.
        
    test : ndarray
        The testing set indices for that split.
        
    scoring : str, callable, dict of strings and callables
        Strategy to evaluate the performance of the estimator. A dictionnary can be used for multiple scores.

    Returns
    -------
    List of two dictionnaries 
        Contains the training and the testing scores associated with the estimator and the train-test split.

    """    
    X_train , y_train , X_test , y_test = X[train , :] , y[train] , X[test , :] , y[test]
    estimator.fit(X_train, y_train)                    
    return _compute_scores(estimator , X_train , y_train , X_test , y_test , scoring)

def _compute_scores(estimator , X_train , y_train , X_test , y_test , scoring):
    """ Given a fitted estimator and a train-test split (X_train , y_train , X_test , y_test),
    compute the trainig scores and the test scores (defined by the scoring parameter) and save 
    them in two separate dictionaries.
    
    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        
    X_train : array-like, shape (n_samples , n_features)
    
    y_train : array-like, shape (n_samples , n_output)

    X_test : array-like, shape (n_samples , n_features)

    y_test : array-like, shape (n_samples , n_output)

    scoring : str, callable, dict of strings and callables
        Strategy to evaluate the performances of the estimator in the outer loop. 
        A dictionnary can be used for multiple scores.

    Returns
    -------
    List of two dictionaries 
    
    """
    training_scores , test_scores = {} , {}
    if isinstance(scoring, dict):
        for k in scoring.keys() : 
            scorer = metrics.check_scoring(estimator , scoring[k])               
            training_scores[k] , test_scores[k] = scorer(estimator, X_train, y_train) , scorer(estimator, X_test, y_test)  
    elif isinstance(scoring, str):
        scorer = metrics.check_scoring(estimator , scoring)               
        training_scores[scoring] , test_scores[scoring] = scorer(estimator, X_train, y_train) , scorer(estimator, X_test, y_test)  
    else :
        scorer = metrics.check_scoring(estimator , scoring)
        training_scores['score'] , test_scores['score'] = scorer(estimator, X_train, y_train) , scorer(estimator, X_test, y_test)      
    return [training_scores , test_scores]

def _aggregate_score_dicts(scores , name = None):
    """ Aggregate a list of dict to a dict of lists.
        
    Parameters
    ----------
    scores : list of dictionaries
        Contains a dictionary of scores for each fold.
        
    name : str, optional
        Prefix for the keys. The default is None.

    Returns
    -------
    dict of lists
    
    Example
    -------      
    
    scores = [{'roc_auc' : 0.78 , 'accuracy' : 0.8} , {'roc_auc' : 0.675 , 'accuracy' : 0.56} , {'roc_auc' : 0.8 , 'accuracy' : 0.72 }]
    
    _aggregate_score_dicts(scores) = {'roc_auc' : [0.78 , 0.675 , 0.8] , 'accuracy' : [0.8 , 0.56 , 0.72]}
    
    """
    if name is None :
        return {key: [score[key] for score in scores] for key in scores[0]}
    else : 
        return {name + str('_') + key: [score[key] for score in scores] for key in scores[0]}
    
def _enumerate(generator , L):
    """ Generator created from a list and another generator which yields two items. 
    
    Parameters
    ----------
    generator : generator (yielding two items)

    L : list
    
    Yields
    ------
    Tuple containing two items from the generator and one element of the list L.
    
    """
    i = 0
    for item1 , item2 in generator:
        yield(item1 , item2 , L[i])
        i += 1
    
def _validate_cv(cv):
    """ Check cross-validation strategy and ensure that the random state is fixed to obtain
    the same partition each time the split method is called.
    
    Parameters
    ----------
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.

    Returns
    -------
    cv_checked : a cross-validator instance.
        The return value is a cross-validator which generates the train/test splits via the split method.
    """
    cv_checked = check_cv(cv = cv)
    try :
       getattr(cv_checked , 'shuffle') 
    except AttributeError :
        if getattr(cv_checked, 'random_state') is None:
            setattr(cv_checked, 'random_state', np.random.randint(0 , 100))
    else : 
        if getattr(cv_checked , 'shuffle') and getattr(cv_checked, 'random_state') is None:
            setattr(cv_checked, 'random_state', np.random.randint(0 , 100))
    return cv_checked