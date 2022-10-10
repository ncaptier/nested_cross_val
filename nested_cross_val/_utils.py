import numpy as np
from sklearn import metrics
from sklearn.model_selection import check_cv


def _fit_out(estimator, X, y, train, test, scoring):
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
    X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
    estimator.fit(X_train, y_train)
    return _compute_scores(estimator, X_train, y_train, X_test, y_test, scoring)


def _compute_scores(estimator, X_train, y_train, X_test, y_test, scoring):
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
    training_scores, test_scores = {}, {}
    if isinstance(scoring, dict):
        for k in scoring.keys():
            scorer = metrics.check_scoring(estimator, scoring[k])
            try:
                training_scores[k], test_scores[k] = scorer(estimator, X_train, y_train), scorer(estimator, X_test,
                                                                                                 y_test)
            except AttributeError:
                training_scores[k], test_scores[k] = np.nan, np.nan
    elif isinstance(scoring, str):
        scorer = metrics.check_scoring(estimator, scoring)
        try:
            training_scores[scoring], test_scores[scoring] = scorer(estimator, X_train, y_train), scorer(estimator,
                                                                                                         X_test, y_test)
        except AttributeError:
            training_scores[scoring], test_scores[scoring] = np.nan, np.nan
    else:
        scorer = metrics.check_scoring(estimator, scoring)
        try:
            training_scores['score'], test_scores['score'] = scorer(estimator, X_train, y_train), scorer(estimator,
                                                                                                         X_test, y_test)
        except AttributeError:
            training_scores['score'], test_scores['score'] = np.nan, np.nan

    return [training_scores, test_scores]


def _aggregate_score_dicts(scores, name=None):
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

    scores = [{'roc_auc' : 0.78 , 'accuracy' : 0.8} , {'roc_auc' : 0.675 , 'accuracy' : 0.56} ,
    {'roc_auc' : 0.8 , 'accuracy' : 0.72 }]

    _aggregate_score_dicts(scores) = {'roc_auc' : [0.78 , 0.675 , 0.8] , 'accuracy' : [0.8 , 0.56 , 0.72]}

    """
    if name is None:
        return {key: [score[key] for score in scores] for key in scores[0]}
    else:
        return {name + str('_') + key: [score[key] for score in scores] for key in scores[0]}


def _enumerate(generator, L):
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
    for item1, item2 in generator:
        yield item1, item2, L[i]
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
    cv_checked = check_cv(cv=cv)
    try:
        getattr(cv_checked, 'shuffle')
    except AttributeError:
        if getattr(cv_checked, 'random_state') is None:
            setattr(cv_checked, 'random_state', np.random.randint(0, 100))
    else:
        if getattr(cv_checked, 'shuffle') and getattr(cv_checked, 'random_state') is None:
            setattr(cv_checked, 'random_state', np.random.randint(0, 100))
    return cv_checked
