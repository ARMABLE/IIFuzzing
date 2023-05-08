import json
import os

import hdbscan

import joblib
import numpy as np
import math
import weka.core.jvm
import weka.core.converters

from UnsupervisedScalableRepresentationLearningTimeSeries import scikit_wrappers

CONFIDENCE = 0.1
STEPS = 40

def load_UEA_dataset(path, dataset):
    """
    Loads the UEA dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # Initialization needed to load a file with Weka wrappers
    weka.core.jvm.start()
    loader = weka.core.converters.Loader(
        classname="weka.core.converters.ArffLoader"
    )

    train_file = os.path.join(path, dataset + "_TRAIN.arff")
    test_file = os.path.join(path, dataset + "_TEST.arff")
    print(train_file, test_file)
    train_weka = loader.load_file(train_file)
    test_weka = loader.load_file(test_file)

    train_size = train_weka.num_instances
    test_size = test_weka.num_instances
    nb_dims = train_weka.get_instance(0).get_relational_value(0).num_instances
    length = train_weka.get_instance(0).get_relational_value(0).num_attributes

    train = np.empty((train_size, nb_dims, length))
    test = np.empty((test_size, nb_dims, length))
    train_labels = np.empty(train_size, dtype=np.int)
    test_labels = np.empty(test_size, dtype=np.int)

    for i in range(train_size):
        train_labels[i] = int(train_weka.get_instance(i).get_value(1))
        time_series = train_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            train[i, j] = time_series.get_instance(j).values

    for i in range(test_size):
        test_labels[i] = int(test_weka.get_instance(i).get_value(1))
        time_series = test_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            test[i, j] = time_series.get_instance(j).values

    # # Normalizing dimensions independently
    # for j in range(nb_dims):
    #     # Post-publication note:
    #     # Using the testing set to normalize might bias the learned network,
    #     # but with a limited impact on the reported results on few datasets.
    #     # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    #     mean = np.nanmean(np.concatenate([train[:, j], test[:, j]]))
    #     var = np.nanvar(np.concatenate([train[:, j], test[:, j]]))
    #     if var != 0:
    #         train[:, j] = (train[:, j] - mean) / math.sqrt(var)
    #         test[:, j] = (test[:, j] - mean) / math.sqrt(var)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train_labels = np.vectorize(transform.get)(train_labels)
    test_labels = np.vectorize(transform.get)(test_labels)

    weka.core.jvm.stop()
    return train, train_labels, test, test_labels


def preprocess(data, nb_dims):
    # Normalizing dimensions independently
    for j in range(nb_dims):
        mean = np.nanmean(data[:, j])
        var = np.nanvar(data[:, j])
        if var != 0:
            data[:, j] = (data[:, j] - mean) / math.sqrt(var)
    return data


def preprocess_(train, test, nb_dims, mean_var):
    # Normalizing dimensions independently
    for j in range(nb_dims):
        # Post-publication note:
        # Using the testing set to normalize might bias the learned network,
        # but with a limited impact on the reported results on few datasets.
        # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
        mean = np.nanmean(np.concatenate([train[:, j], test[:, j]]))
        var = np.nanvar(np.concatenate([train[:, j], test[:, j]]))
        # mean, var = mean_var[j]
        if var != 0:
            train[:, j] = (train[:, j] - mean) / math.sqrt(var)
            test[:, j] = (test[:, j] - mean) / math.sqrt(var)
    return train, test


def load_model(save_path, dataset):
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()
    hf = open(
        os.path.join(
            save_path, dataset + '_hyperparameters.json'
        ), 'r'
    )
    hp_dict = json.load(hf)
    hf.close()
    hp_dict['cuda'] = True
    hp_dict['gpu'] = 0
    classifier.set_params(**hp_dict)
    classifier.load(os.path.join(save_path, dataset))
    return classifier


def deep_gini(pred_test_prob):
    metrics = np.array(np.sum(pred_test_prob**2, axis=1))
    # print(metrics.shape)
    # rank_list = np.argsort(metrics)
    # print(rank_list)
    # return rank_list
    return metrics


def predict(encoder, clusterer, points_to_predict):
    test_features = encoder.encode(points_to_predict, batch_size=50)
    # print(data.shape, test_features.shape)

    # y_test_pred, test_prob = hdbscan.approximate_predict(clusterer, test_features)
    # test_prob = hdbscan.approximate_predict_scores(clusterer, test_features)
    membership_vectors = hdbscan.membership_vector(clusterer, test_features)
    # print(membership_vectors)
    gini_score = deep_gini(membership_vectors)
    pred_labels = np.argmax(membership_vectors, axis=1)
    pred_labels[gini_score < CONFIDENCE] = -1
    # print(hdbscan.approximate_predict_scores(cluster_model, test_features))
    return pred_labels, gini_score


if __name__== "__main__" :
    dataset = 'RL_CARLA'
    best_step = STEPS
    data_path = '../data/TestData'
    model_path = '../models'
    encoder = load_model(os.path.join(model_path, 'rl_model'), dataset)
    clusterer = joblib.load(os.path.join(model_path, 'cluster_model', dataset, dataset + '_' + str(best_step) + '.pkl'))
    train, train_label, test, test_label = load_UEA_dataset(os.path.join(data_path, dataset, 'Test'), dataset)
    train, test = train[:, :, :best_step], test[:, :, :best_step]
    print(train.shape, test.shape)
    # train, test = preprocess_(train, test, train.shape[1], mean_var)
    # train = preprocess(train, train.shape[1])
    # test = preprocess(test, test.shape[1])

    for data in train:
        data = data.reshape((1, -1, best_step))
        data = preprocess(data, data.shape[1])
        l, p = predict(encoder, clusterer, data)
        print(l, p)
