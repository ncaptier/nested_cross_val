import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve

"""
This module implements performance metrics for binary classification based on the Matthews correlation coefficient. We advise 
the user to read the two following articles for more details :
    - https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7
    - https://arxiv.org/abs/2006.11278
"""


def nmcc_from_confusion(tp, fp, fn, tn):
    """ Compute the normalized Matthews correlation coefficient from a confusion matrix.

    Parameters
    ----------
    tp : int
        Number of true positives

    fp : int
        Number of false positives

    fn : int
        Number of false negatives

    tn : int
        Number of true negatives

    Returns
    -------
    Float between 0 and 1
        Normalized Matthews correlation coefficient.

    """
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.5
    else:
        return 0.5 * ((tp * tn - fp * fn) / np.sqrt(denom) + 1)


def f1_from_confusion(tp, fp, fn, tn):
    """ Compute the F1 score from a confusion matrix.

    Parameters
    ----------
    tp : int
        Number of true positives

    fp : int
        Number of false positives

    fn : int
        Number of false negatives

    tn : int
        Number of true negatives

    Returns
    -------
    f1 : Float between 0 and 1
        F1 score.

    """
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1


def mcc_f1_curve(y_true, y_score):
    """ Compute the MCC- F1 curve


    Parameters
    ----------
    y_true : ndarray of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    Returns
    -------
    f1s : TYPE
        F1 scores for different thresholds

    nmccs : TYPE
        Normalized Matthews correlation coefficients for different thresholds

    thresholds : TYPE
        Decreasing thresholds on y_score used to compute F1 score and Matthews correlation coefficient.

    Note
    -----

    Please refer to https://arxiv.org/abs/2006.11278 for all the details.

    """
    N, P = np.sum(y_true == 0), np.sum(y_true == 1)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    fns = P - tps
    tns = N - fps

    nmccs = np.vectorize(nmcc_from_confusion)(tps, fps, fns, tns)
    f1s = f1_from_confusion(tps, fps, fns, tns)

    return f1s, nmccs, thresholds


def mcc_f1_score(y_true, y_score, bins=100):
    """ Compute the MCC-F1 score associated with the MCC-F1 curve

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    bins : int, optional
        Number of sub-ranges for the full range of normalized. The default is 100.

    Returns
    -------
    metric : float between 0 and 1
        MCC-F1 metric to assess the performance of the classifier.

    best_thresholds : float
        Threshold that provides the best prediction performance.

    Note
    -----

    Please refer to https://arxiv.org/abs/2006.11278 for all the details.

    """

    f1s, nmccs, thresholds = mcc_f1_curve(y_true, y_score)
    distances = np.sqrt((1 - f1s) ** 2 + (1 - nmccs) ** 2
                        )
    idmax = np.argmax(nmccs)
    left_nmccs, left_d = nmccs[:idmax + 1], distances[: idmax + 1]
    right_nmccs, right_d = nmccs[idmax + 1:], distances[idmax + 1:]

    minimum, maximum = np.min(nmccs), np.max(nmccs)
    w = (maximum - minimum) / bins

    # Initialisation
    idx_L = np.argwhere((left_nmccs >= minimum) * (left_nmccs <= minimum + w))
    idx_R = np.argwhere((right_nmccs >= minimum) * (right_nmccs <= minimum + w))
    L = [np.mean(left_d[idx_L])] if len(idx_L) > 0 else []
    R = [np.mean(right_d[idx_R])] if len(idx_R) > 0 else []

    for i in range(1, bins):
        idx_L = np.argwhere((left_nmccs > minimum + i * w) * (left_nmccs <= minimum + (i + 1) * w))
        idx_R = np.argwhere((right_nmccs > minimum + i * w) * (right_nmccs <= minimum + (i + 1) * w))
        if len(idx_L) > 0:
            L.append(np.mean(left_d[idx_L]))
        if len(idx_R) > 0:
            R.append(np.mean(right_d[idx_R]))

    metric = 1 - (np.mean(L + R) / np.sqrt(2))
    best_thresholds = thresholds[np.argmin(distances)]

    return metric, best_thresholds
