from sklearn import metrics
from transformers import BertForSequenceClassification, BertTokenizer
import config
import numpy as np
import scipy
import json
import pprint

def hamming_score(y_true, y_pred, normalize=True,
                    sample_weight=None):
    # https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def get_multilabel_metrics(scores, labels):
    squished = th.sigmoid(th.from_numpy(scores))
    pred = squished
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    hamming_loss = metrics.hamming_loss(labels, pred)
    h_score = hamming_score(np.array(labels), np.array(pred))

    return {
        'Hamming_loss': hamming_loss,
        'Hamming_score': h_score
    }


def update_metrics(metrics, metric_file):
    filepath = f'{metric_file}.json'
    try:
        with open(filepath) as f:
            old_metrics = json.load(f)
        for key in metrics:
            if key in old_metrics:
                old_list = old_metrics[key]
                old_list.append(metrics[key])
                old_metrics[key] = old_list
            else:
                old_metrics[key] = [metrics[key]]
        with open(filepath, 'w') as f:
            json.dump(old_metrics, f)
    except FileNotFoundError:
        if metrics is not None:
            jsondict = dict()
            for key in metrics:
                jsondict[key] = [metrics[key]]
            with open(filepath, 'w') as f:
                json.dump(jsondict, f)
        else:
            return


def get_binary_metrics(scores, labels, out_file):
    """
    scores: np.array (N,2)
    labels: np.array (N,)
    """
    pred = np.argmax(scores, axis=1)
    accuracy = metrics.accuracy_score(labels, pred)
    f1_score = metrics.f1_score(labels, pred)
    labels_unflattened = np.zeros_like(scores)
    labels_unflattened[labels == 0] = [1, 0]
    labels_unflattened[labels == 1] = [0, 1]
    roc_auc = metrics.roc_auc_score(labels_unflattened, scores)
    conf_mat = metrics.confusion_matrix(labels, pred)
    sensitvity = conf_mat[1, 1]/(conf_mat[1, 1]+conf_mat[1, 0])
    specificity = conf_mat[0, 0]/(conf_mat[0, 0]+conf_mat[0, 1])
    pr_auc = metrics.average_precision_score(labels_unflattened, scores)
    metric_dict = {
        'Accuracy': accuracy,
        'F1_score': f1_score,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Specificity': specificity,
        'Sensitivity': sensitvity,
        'G-Mean': np.sqrt(sensitvity*specificity),
        'Conf_Mat': conf_mat.tolist()
        }
    update_metrics(metric_dict, out_file)
    print(metric_dict)
    del metric_dict['Conf_Mat']
    return metric_dict


if __name__ == '__main__':
    update_metrics('out/metrics.json')