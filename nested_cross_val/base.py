import dask
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.model_selection import RandomizedSearchCV as dask_RandomizedSearchCV
from sklearn.base import clone, check_array, BaseEstimator, MetaEstimatorMixin
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.exceptions import NotFittedError
# from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from ._utils import _aggregate_score_dicts, _enumerate, _validate_cv, _fit_out, _compute_scores, CustomStackingRegressor

# from sksurv.linear_model import CoxPHSurvivalAnalysis


class NestedCV(BaseEstimator, MetaEstimatorMixin):
    """ Implement a nested cross-validation pipeline compatible with sklearn API.

    Parameters
    ----------

    estimators : dict.
        Dictionary of estimators compatible with sklearn API. Each estimator will be fitted with nested cross-validation
        (same folds) (e.g. {name_estimator1: estimator1, name_estimator2: estimator2}.

    params : dict.
        Nested dictionary with the same keys as estimator. Each estimator is associated with a dictionary of
        hyperparameters to optimize (see sklearn GridSearchCV and RandomizedSearchCV for more details).

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
        Strategy to evaluate the performance of the model in the inner loop.
        See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for more details.

    scoring_outer : str, callable, dict of strings and callables
        Strategy to evaluate the performances of the model in the outer loop. A dictionnary can be used for
        multiple scores.

    ensembling_method : int, {'stacking_classifier', 'stacking_regressor', 'hard_voting_classifier',
    'soft_voting_regressor', 'voting_regressor'}

    voting_weights :

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

    scheduler : string, callable, Client, or None, optionnal.
        The dask scheduler to use.
        See https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.html for more details.
        Only available when library = "dask".
        The default is None.

    verbose : int
        Controls the verbosity: the higher, the more messages. The computation time for each inner fold is displayed
        when verbose > 0.

    Attributes
    ----------

    best_params_ : list
        Params settings that were selected by cross-validation for each outer train-test split.

    best_estimators_ : list
        Estimators that were selected by cross-validation. If refit_estimators is True each one is refitted on the full
        training data (i.e. with its associated outer train-test split).

    inner_results_ : dict
        Nested dictionary that saves all the results for each inner cross-validation
       (one for each outer train-test split).

    outer_results_ : dict

    X_ : array-like, shape (n_samples , n_features)
        Save the training data for additional computations once the method is fitted.

    y_ : array-like, shape (n_samples , n_output)
        Save the training data for additional computations once the method is fitted.

    cv_outer_ : cross-validation generator
        Save the outer cross-validation shceme (with fixed random state) for additional computations once the method is
        fitted

    Notes
    -----

    Dask_ML optimizes computational time and memory during hyperparameter optimization
    See https://ml.dask.org/hyper-parameter-search.html and
    https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.htm for more details

    """

    def __init__(self, estimators, params, cv_inner, cv_outer, scoring_inner, scoring_outer, ensembling_method=None,
                 voting_weights=None, n_jobs_inner=-1, n_jobs_outer=-1, refit_estimators=True, randomized=False,
                 scheduler=None, stacking_estimator=None, verbose=1):

        self.estimators = estimators
        self.params = params
        self.n_jobs_inner = n_jobs_inner
        self.n_jobs_outer = n_jobs_outer
        self.refit_estimators = refit_estimators
        self.randomized = randomized
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer
        self.scoring_inner = scoring_inner
        self.scoring_outer = scoring_outer
        self.ensembling_method = ensembling_method
        self.voting_weights = voting_weights
        self.scheduler = scheduler
        self.verbose = verbose
        self.stacking_estimator = stacking_estimator

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

    def _inner_dask(self, X, y):
        """ Perform hyperparameter tuning with cross-validation for each outer fold (using dask-ml library).
        Each time, save best parameters and associated best estimator (not refitted !).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data set.

        y : array-like, shape (n_samples, n_output)
            Target relative to X for classification or regression.
        """

        # 1. Set up weights in case of voting strategy for ensembling
        if self.voting_weights == 'auto':
            self.voting_weights_ = [[] for _ in range(self.cv_outer_.get_n_splits())]
        else:
            self.voting_weights_ = self.voting_weights

        # 2. Perform hyperparameter tuning for each outer fold and each estimator
        for estim in self.estimators.keys():

            cv_estimator = clone(self.estimators[estim])

            if self.randomized:
                inner = dask_RandomizedSearchCV(estimator=cv_estimator, param_distributions=self.params[estim],
                                                scoring=self.scoring_inner, return_train_score=False
                                                , cv=self.cv_inner_, n_jobs=self.n_jobs_inner, refit=False,
                                                scheduler=self.scheduler)
            else:
                inner = dask_GridSearchCV(estimator=cv_estimator, param_grid=self.params[estim],
                                          scoring=self.scoring_inner, return_train_score=False
                                          , cv=self.cv_inner_, n_jobs=self.n_jobs_inner, refit=False,
                                          scheduler=self.scheduler)

            count = 0
            for train, _ in tqdm(self.cv_outer_.split(X, y), total=self.cv_outer_.get_n_splits(),
                                 desc='Tuning ' + estim, disable=self._disable_tqdm):
                X_train, y_train = X[train, :], y[train]
                inner.fit(X_train, y_train)
                self.inner_results_[estim]['out fold ' + str(count)] = inner.cv_results_

                # best_params_ attribute is not provided with dask-ml method when refit = False, we need to extract it manually
                best_params_ = inner.cv_results_["params"][inner.cv_results_["rank_test_score"].argmin()]
                self.best_params_[estim].append(best_params_)
                self.best_estimators_[estim].append(clone(clone(self.estimators[estim]).set_params(**best_params_)))

                if self.voting_weights == 'auto':
                    self.voting_weights_[count].append(
                        inner.cv_results_["mean_test_score"][inner.cv_results_["rank_test_score"].argmin()])
                count += 1

        return self

    def _outer_with_fitted(self, fitted_estimators):
        """

        Parameters
        ----------
        fitted_estimators : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        for estim in tqdm(self.estimators.keys(), desc="Test best estimators", disable=self._disable_tqdm):
            outputs = []
            for (train, test, estimator) in _enumerate(self.cv_outer_.split(self.X_, self.y_),
                                                       fitted_estimators[estim]):
                out = dask.delayed(_compute_scores)(estimator, self.X_[train, :], self.y_[train], self.X_[test, :],
                                                    self.y_[test], self.scoring_outer)
                outputs.append(out)

            L = list(zip(*dask.compute(*outputs)))

            self.outer_results_[estim].update(_aggregate_score_dicts(L[0], name='training'))
            self.outer_results_[estim].update(_aggregate_score_dicts(L[1], name='test'))
        return self

    def _outer_with_unfitted(self):
        """

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        for estim in tqdm(self.estimators.keys(), desc="Test best estimators", disable=self._disable_tqdm):
            outputs = []
            if self.refit_estimators:
                for (train, test, estimator) in _enumerate(self.cv_outer_.split(self.X_, self.y_),
                                                           self.best_estimators_[estim]):
                    out = dask.delayed(_fit_out)(estimator, self.X_, self.y_, train, test, self.scoring_outer)
                    outputs.append(out)
            else:
                for (train, test, estimator) in _enumerate(self.cv_outer_.split(self.X_, self.y_),
                                                           self.best_estimators_[estim]):
                    out = dask.delayed(_fit_out)(clone(estimator), self.X_, self.y_, train, test, self.scoring_outer)
                    outputs.append(out)

            L = list(zip(*dask.compute(*outputs)))

            self.outer_results_[estim].update(_aggregate_score_dicts(L[0], name='training'))
            self.outer_results_[estim].update(_aggregate_score_dicts(L[1], name='test'))

        return self

    def fit(self, X, y):
        """ Fit the nested cross-validation with X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data set.

        y : array-like, shape (n_samples, n_output)
            Target relative to X for classification or regression.
        """

        # 1. Initialisation

        self.X_ = check_array(X, force_all_finite="allow-nan")
        self.y_ = check_array(y, ensure_2d=False, dtype=None)
        self.cv_inner_ = _validate_cv(self.cv_inner)
        self.cv_outer_ = _validate_cv(self.cv_outer)

        self.inner_results_ = {key: {} for key in self.estimators.keys()}
        self.outer_results_ = {key: {} for key in self.estimators.keys()}
        self.best_params_ = {key: [] for key in self.estimators.keys()}
        self.best_estimators_ = {key: [] for key in self.estimators.keys()}

        if self.verbose <= 0:
            self._disable_tqdm = True
        else:
            self._disable_tqdm = False

        # 2. Inner loops (tuning by cross-validation)

        self._inner_dask(self.X_, self.y_)

        # 3. Ensembling and outer loop (cross-validation)

        if self.ensembling_method is not None:
            fitted_estimators = self._ensembling(self.X_, self.y_)
            self._outer_with_fitted(fitted_estimators=fitted_estimators)

        else:
            self._outer_with_unfitted()

        return self

    def _ensembling(self, X, y):

        ensemble_estimators = []
        s = []
        i = 0
        if self.ensembling_method == 'stacking_classifier':
            for train, test in tqdm(self.cv_outer_.split(X, y), total=self.cv_outer_.get_n_splits(),
                                    desc='Fitting ' + self.ensembling_method, disable=self._disable_tqdm):
                X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
                ensembling = StackingClassifier(
                    estimators=[(key, self.best_estimators_[key][i]) for key in self.best_estimators_.keys()],
                    final_estimator=self.stacking_estimator,
                    cv=self.cv_inner_,
                    stack_method='auto',
                    n_jobs=-1)
                ensembling.fit(X_train, y_train)
                ensemble_estimators.append(ensembling)
                s.append(_compute_scores(ensembling, X_train, y_train, X_test, y_test, self.scoring_outer))
                i += 1

        elif self.ensembling_method == 'stacking_regressor':
            for train, test in tqdm(self.cv_outer_.split(X, y), total=self.cv_outer_.get_n_splits(),
                                    desc='Fitting ' + self.ensembling_method, disable=self._disable_tqdm):
                X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
                ensembling = CustomStackingRegressor(
                    estimators=[(key, self.best_estimators_[key][i]) for key in self.best_estimators_.keys()],
                    final_estimator=self.stacking_estimator,
                    cv=self.cv_inner_,
                    n_jobs=-1)
                ensembling.fit(X_train, y_train)
                ensemble_estimators.append(ensembling)
                s.append(_compute_scores(ensembling, X_train, y_train, X_test, y_test, self.scoring_outer))
                i += 1

        elif self.ensembling_method == 'hard_voting_classifier':
            for train, test in tqdm(self.cv_outer_.split(X, y), total=self.cv_outer_.get_n_splits(),
                                    desc='Fitting ' + self.ensembling_method, disable=self._disable_tqdm):
                X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
                ensembling = VotingClassifier(
                    estimators=[(key, self.best_estimators_[key][i]) for key in self.best_estimators_.keys()],
                    weights=self.voting_weights_[i],
                    n_jobs=-1)
                ensembling.fit(X_train, y_train)
                ensemble_estimators.append(ensembling)
                s.append(_compute_scores(ensembling, X_train, y_train, X_test, y_test, self.scoring_outer))
                i += 1

        elif self.ensembling_method == 'soft_voting_classifier':
            for train, test in tqdm(self.cv_outer_.split(X, y), total=self.cv_outer_.get_n_splits(),
                                    desc='Fitting ' + self.ensembling_method, disable=self._disable_tqdm):
                X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
                ensembling = VotingClassifier(
                    estimators=[(key, self.best_estimators_[key][i]) for key in self.best_estimators_.keys()],
                    voting='soft',
                    weights=self.voting_weights_[i],
                    n_jobs=-1)
                ensembling.fit(X_train, y_train)
                ensemble_estimators.append(ensembling)
                s.append(_compute_scores(ensembling, X_train, y_train, X_test, y_test, self.scoring_outer))
                i += 1

        elif self.ensembling_method == 'voting_regressor':
            for train, test in tqdm(self.cv_outer_.split(X, y), total=self.cv_outer_.get_n_splits(),
                                    desc='Fitting ' + self.ensembling_method, disable=self._disable_tqdm):
                X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
                ensembling = VotingRegressor(
                    estimators=[(key, self.best_estimators_[key][i]) for key in self.best_estimators_.keys()],
                    weights=self.voting_weights_,
                    n_jobs=-1)
                ensembling.fit(X_train, y_train)
                ensemble_estimators.append(ensembling)
                s.append(_compute_scores(ensembling, X_train, y_train, X_test, y_test, self.scoring_outer))
                i += 1

        else:
            raise ValueError()

        L = list(zip(*s))
        self.outer_results_[self.ensembling_method] = _aggregate_score_dicts(L[0], name='training')
        self.outer_results_[self.ensembling_method].update(_aggregate_score_dicts(L[1], name='test'))

        fitted_estimators = {}
        for key in self.best_estimators_.keys():
            fitted_estimators[key] = [ens.named_estimators_[key] for ens in ensemble_estimators]

        if self.refit_estimators:
            self.best_estimators_ = fitted_estimators
            self.best_estimators_[self.ensembling_method] = ensemble_estimators

        return fitted_estimators

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

        for estim in self.estimators.keys():
            i = 0
            s = []
            for train, test in self.cv_outer_.split(self.X_, self.y_):
                X_train, y_train, X_test, y_test = self.X_[train, :], self.y_[train], self.X_[test, :], self.y_[test]
                estimator = self.best_estimators_[estim][i]
                s.append(_compute_scores(estimator, X_train, y_train, X_test, y_test, scoring))
                i += 1
            L = list(zip(*s))
            self.outer_results_[estim].update(_aggregate_score_dicts(L[0], name='training'))
            self.outer_results_[estim].update(_aggregate_score_dicts(L[1], name='test'))
        return

    def get_predictions(self, estimator):
        """ Collect predictions for each estimator in best_estimators_

        Only available if refit_estimators = True.

        Yields
        -------
        (array of shape (n_samples) or (n_samples , n_outputs) , array of shape (n_samples) or (n_samples , n_outputs))
            Tuple containing y_pred and y for each train-test split defined by cv_outer_.split(X_ , y_)

        """
        self._check_is_fitted('get_predictions')
        i = 0
        for _, test in self.cv_outer_.split(self.X_, self.y_):
            estim = self.best_estimators_[estimator][i]
            yield estim.predict(self.X_[test, :]), self.y_[test]
            i += 1

    def get_probas(self, estimator, method='predict_proba'):
        """ Collect confidence values for each estimator in best_estimators_

        Only available is refit_estimators = True.

        Parameters
        ----------

        estimator :

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
        for _, test in self.cv_outer_.split(self.X_, self.y_):
            estim = self.best_estimators_[estimator][i]
            probas = getattr(estim, method)(self.X_[test, :])
            if len(probas.shape) == 2 and probas.shape[1] == 2:
                probas = probas[:, 1]
            yield probas, self.y_[test]
            i += 1

    def get_attributes(self, estimator, attribute):
        """ Collect attribute for each estimator in best_estimators_.

        Only available is refit_estimators = True.

        Parameters
        ----------

        estimator :

        attribute : str or callable
            If attribute is a string, it should correspond to a valid attribute of the estimators
            contained in self.best_estimators_.

        Yields
        ------
        Any
            object defined by the parameter "attribute".

        """
        self._check_is_fitted('get_attributes')

        for estim in self.best_estimators_[estimator]:
            if isinstance(attribute, str):
                yield getattr(estim, attribute)
            else:
                yield attribute(estim)
