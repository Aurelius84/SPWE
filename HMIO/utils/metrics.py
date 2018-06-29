"""
Metrics for multi-label classification.
"""
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, label_ranking_loss, average_precision_score, coverage_error
from .base_metrics import _hamming_loss, _one_error, _average_precision, _coverage, _ranking_loss


def F1_measure(labels, preds, average='binary', mode=1):
    '''
    Compute the scores for each output separately
    support 'binary' (default), 'micro', 'macro', 'samples'
    '''
    f1_scores = float(f1_score(labels, preds, average=average))
    return f1_scores


def Hamming_loss(labels, preds, mode=1):
    '''
    用于度量样本在单个标记上的真实标记和预测标记的错误匹配情况
    @labels: true labels of samples
    @preds:  predict labels of samples
    '''
    if mode:
        hl = hamming_loss(labels, preds)
    else:
        hl = np.mean(list(map(_hamming_loss, preds, labels)))
    return hl


def One_error(labels, probs, mode=1):
    '''
    用来考察预测值排在第一位的标记却不隶属于该样本的情况
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    if mode:
        idx = probs.argsort(axis=1)[:, -1:]
        targets_top_1 = [labels[i, idx[i][0]] for i in range(len(idx))]
        error = 1. - np.mean(targets_top_1)
    else:
        error = 1 - np.mean(list(map(_one_error, probs, labels)))

    return error


def Ranking_loss(labels, probs, mode=1):
    '''
    用来考察样本的不相关标记的排序低于相关标记的排序情况
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    if mode:
        rl = label_ranking_loss(labels, probs)
    else:
        rl = np.mean(list(map(_ranking_loss, probs, labels)))

    return rl


def Average_precision(labels, probs, mode=1,average="samples"):
    '''
    用来考察排在隶属于该样本标记之前标记仍属于样本的相关标记集合的情况
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    if mode:
        ap = average_precision_score(labels, probs, average=average)
    else:
        ap = np.mean(list(map(_average_precision, probs, labels)))

    return ap


def Coverage(labels, probs, mode=1):
    '''
    用于度量平均上需要多少步才能遍历样本所有的相关标记
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    if mode:
        steps = coverage_error(labels, probs)
    else:
        steps = np.mean(list(map(_coverage, probs, labels)))

    return steps


def Construct_thresholds(targets, probs, top_k=None):
    '''
    构建最优阈值
    :param targets:
    :param probs:
    :param top_k:
    :return:
    '''
    targets = np.array(targets)
    probs = np.array(probs)
    assert probs.shape == targets.shape, \
        "The shape of predictions should match the shape of targets."
    nb_samples, nb_labels = targets.shape
    top_k = top_k or nb_labels

    # Sort predicted probabilities in descending order
    idx = np.argsort(probs, axis=1)[:, :-(top_k + 1):-1]
    p_sorted = np.vstack([probs[i, idx[i]] for i in range(len(idx))])
    t_sorted = np.vstack([targets[i, idx[i]] for i in range(len(idx))])

    # Compute F-1 measures for every possible threshold position
    F1 = []
    TP = np.zeros(nb_samples)
    FN = t_sorted.sum(axis=1)
    FP = np.zeros(nb_samples)
    for i in range(top_k):
        TP += t_sorted[:, i]
        FN -= t_sorted[:, i]
        FP += 1 - t_sorted[:, i]
        F1.append(2 * TP / (2 * TP + FN + FP))
    F1 = np.vstack(F1).T
    # Find the thresholds
    row = np.arange(nb_samples)
    col = F1.argmax(axis=1)
    p_sorted = np.hstack([p_sorted, np.zeros(nb_samples)[:, None]])
    # T = p_sorted[row, col][:,None]
    ratio = 0.42
    T = (ratio * p_sorted[row, col] +
         (1 - ratio) * p_sorted[row, col + 1])[:, None]

    return T


if __name__ == '__main__':

    y_true = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
    y_score = np.array([[1, 0.5, 0.75], [0.1, 0.8, 1], [0.3, 0.7, 0.8]])
    y_preds = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

    print('F1@micro: {}'.format(F1_measure(y_true, y_preds, average='micro')))
    print('F1@macro: {}'.format(F1_measure(y_true, y_preds, average='macro')))
    print('hamming_loss: {}, {}'.format(Hamming_loss(y_true, y_preds, mode=1), Hamming_loss(y_true, y_preds, mode=0)))
    print('ranking_loss: {}, {}'.format(Ranking_loss(y_true, y_score, mode=1), Ranking_loss(y_true, y_score, mode=0)))
    print('one_error: {}, {}'.format(One_error(y_true, y_score, mode=1), One_error(y_true, y_score, mode=0)))
    print('coverage: {}, {}'.format(Coverage(y_true, y_score, mode=1), Coverage(y_true, y_score, mode=0)))
    print('average_precision: {}, {}'.format(Average_precision(y_true, y_score, mode=1), Average_precision(y_true, y_score, mode=0)))
    T = Construct_thresholds(y_true, y_score)
    print('best threshold : {}'.format(T))
    print('labels over threshold: {}'.format(y_score>T))
