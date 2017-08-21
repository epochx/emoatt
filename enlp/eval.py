#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import sys
import scipy.stats

class ConfusionMatrix(object):

    def __init__(self, label, true_positives=None, false_positives=None, true_negatives=None,
                 false_negatives=None):
        self.label = label
        self.tp = self.true_positives = true_positives if true_positives else []
        self.fp = self.false_positives = false_positives if false_positives else []
        self.tn = self.true_negatives = true_negatives if true_negatives else []
        self.fn = self.false_negatives = false_negatives if false_negatives else[]

    def __repr__(self, *args, **kwargs):
        string = ''
        string += "True Positives: " + str(len(self.tp)) + '\n'
        string += "False Positives: " + str(len(self.fp)) + '\n'
        string += "True Negatives: " + str(len(self.tn)) + '\n'
        string += "False negatives: " + str(len(self.fn)) + '\n'
        return string

    @property
    def precision(self):
        try:
            return 1.0 * len(self.tp) / (len(self.tp) + len(self.fp))
        except ZeroDivisionError:
            return 0

    @property
    def recall(self):
        try:
            return 1.0 * len(self.tp) / (len(self.tp) + len(self.fn))
        except ZeroDivisionError:
            return 0

    @property
    def fmeasure(self):
        try:
            precision = self.precision
            recall = self.recall
            return 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            return 0

    @property
    def accuracy(self):
        try:
            return 1.0 * (len(self.tp) + len(self.tn)) / (len(self.tp) + len(self.tn) + len(self.fn) + len(self.fp))
        except ZeroDivisionError:
            return 0

    @property
    def measures(self):
        return {"p": self.precision,
                "r": self.recall,
                "f1": self.fmeasure}

    p = precision
    r = recall
    a = accuracy
    f = fmeasure


def classeval(predicted_labels, reference_labels, labels, classification_out_file=None):
    assert len(predicted_labels) == len(reference_labels)

    cm = {}
    for label in labels:
        cm[label] = ConfusionMatrix(label)

    cm["all"] = ConfusionMatrix("all")

    for i, (pred_label, ref_label) in enumerate(zip(predicted_labels, reference_labels)):
        if pred_label == ref_label:
            cm[ref_label].tp.append(i)
            cm["all"].tp.append(i)
            for label in labels:
                if label != ref_label:
                    cm[label].tn.append(i)
        else:
            cm[ref_label].fn.append(i)
            cm[pred_label].fp.append(i)
            cm["all"].fp.append(i)

    if classification_out_file:
        with open(classification_out_file, "w") as f:
            for pred_label, ref_label in zip(predicted_labels, reference_labels):
                f.write("{0} {1}\n".format(pred_label, ref_label))

    return {key: value.measures for key, value in cm.items()}


def regeval(words_list, hyp_values, ref_values, ref_labels, ids, regression_out_file):
    #  id[tab]tweet[tab]emotion[tab]score
    tweets = [ " ".join(tweet_list) for tweet_list in words_list]
    ref_file_path = regression_out_file + ".ref"
    hyp_file_path = regression_out_file + ".hyp"

    with open(hyp_file_path, "w") as f:
        for (idx, tweet, ref_label, hyp_value) in zip(ids, tweets, ref_labels, hyp_values):
            string = "\t".join((idx, tweet, ref_label, str(hyp_value[0]))) + "\n"
            f.write(string.encode("utf-8"))

    if ref_values:
        with open(ref_file_path, "w") as f:
            for (idx, tweet, ref_label, ref_value) in zip(ids, tweets, ref_labels, ref_values):
                string = "\t".join((idx, tweet, ref_label, str(ref_value))) + "\n"
                f.write(string.encode("utf-8"))


        result = correval(hyp_file_path, ref_file_path)

        pear_results = []
        spear_results = []

        pear_results_range_05_1 = []
        spear_results_range_05_1 = []

        pear_results.append(result[0])
        spear_results.append(result[1])

        pear_results_range_05_1.append(result[2])
        spear_results_range_05_1.append(result[3])

        avg_pear = numpy.mean(pear_results)
        avg_spear = numpy.mean(spear_results)

        avg_pear_range_05_1 = numpy.mean(pear_results_range_05_1)
        avg_spear_range_05_1 = numpy.mean(spear_results_range_05_1)


        return avg_pear, avg_spear, avg_pear_range_05_1, avg_spear_range_05_1

    else:
        return 0, 0, 0, 0

def correval(pred, gold):
    f = open(pred, "rb")
    pred_lines = f.readlines()
    f.close()

    f = open(gold, "rb")
    gold_lines = f.readlines()
    f.close()

    if (len(pred_lines) == len(gold_lines)):
        # align tweets ids with gold scores and predictions
        data_dic = {}

        for line in gold_lines:
            parts = line.split('\t')
            if len(parts) == 4:
                data_dic[int(parts[0])] = [float(line.split('\t')[3])]
            else:
                raise ValueError('Format problem.')

        for line in pred_lines:
            parts = line.split('\t')
            if len(parts) == 4:
                if int(parts[0]) in data_dic:
                    try:
                        data_dic[int(parts[0])].append(float(line.split('\t')[3]))
                    except ValueError:
                        # Invalid predictions are replaced by a default value
                        data_dic[int(parts[0])].append(0.5)
                else:
                    raise ValueError('Invalid tweet id.')
            else:
                raise ValueError('Format problem.')

        # lists storing gold and prediction scores
        gold_scores = []
        pred_scores = []

        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_range_05_1 = []
        pred_scores_range_05_1 = []

        for id in data_dic:
            if (len(data_dic[id]) == 2):
                gold_scores.append(data_dic[id][0])
                pred_scores.append(data_dic[id][1])
                if (data_dic[id][0] >= 0.5):
                    gold_scores_range_05_1.append(data_dic[id][0])
                    pred_scores_range_05_1.append(data_dic[id][1])
            else:
                raise ValueError('Repeated id in test data.')

        # return zero correlation if predictions are constant
        if numpy.std(pred_scores) == 0 or numpy.std(gold_scores) == 0:
            return (0, 0, 0, 0)

        pears_corr = scipy.stats.pearsonr(pred_scores, gold_scores)[0]
        spear_corr = scipy.stats.spearmanr(pred_scores, gold_scores)[0]

        pears_corr_range_05_1 = scipy.stats.pearsonr(pred_scores_range_05_1, gold_scores_range_05_1)[0]
        spear_corr_range_05_1 = scipy.stats.spearmanr(pred_scores_range_05_1, gold_scores_range_05_1)[0]

        return (pears_corr, spear_corr, pears_corr_range_05_1, spear_corr_range_05_1)

    else:
        raise ValueError('Predictions and gold data have different number of lines.')




