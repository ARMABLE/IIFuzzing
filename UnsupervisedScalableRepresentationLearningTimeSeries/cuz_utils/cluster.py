import json
import os
import math

from collections import Counter

import hdbscan

import joblib
import numpy as np
import weka.core.jvm
import weka.core.converters

import scikit_wrappers


CONFIDENCE = 0.1


def load_UEA_dataset(path, dataset, best_step):
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

    train_file = os.path.join(path, dataset, dataset + "_TRAIN.arff")
    test_file = os.path.join(path, dataset, dataset + "_TEST.arff")
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

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train_labels = np.vectorize(transform.get)(train_labels)
    test_labels = np.vectorize(transform.get)(test_labels)

    weka.core.jvm.stop()
    return train, train_labels, test, test_labels

def preprocess(train, test, nb_dims):
    # Normalizing dimensions independently
    for j in range(nb_dims):
        # Post-publication note:
        # Using the testing set to normalize might bias the learned network,
        # but with a limited impact on the reported results on few datasets.
        # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
        mean = np.nanmean(np.concatenate([train[:, j], test[:, j]]))
        var = np.nanvar(np.concatenate([train[:, j], test[:, j]]))
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



def _hdbscan(data):
    X = np.array(data)
    hdbscan_ = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=10, cluster_selection_epsilon=1,
                               prediction_data=True).fit(X)
    return hdbscan_


def cluster(data):
    return _hdbscan(data)


def deep_gini(pred_test_prob):
    metrics = np.array(np.sum(pred_test_prob**2, axis=1))
    # print(metrics.shape)
    # rank_list = np.argsort(metrics)
    # print(rank_list)
    # return rank_list
    return metrics


def assign_cluster(clusterer, points_to_predict):
    membership_vectors = hdbscan.membership_vector(clusterer, points_to_predict)
    gini_score = deep_gini(membership_vectors)
    pred_labels = np.argmax(membership_vectors, axis=1)
    pred_labels[gini_score < CONFIDENCE] = -1
    # print(hdbscan.approximate_predict_scores(cluster_model, test_features))
    return pred_labels, gini_score


if __name__== "__main__" :
    dataset = 'RL_CARLA'
    best_step = 40
    data_path = '../data/TestData'
    model_path = '../models'

    train, train_labels, test, test_labels = load_UEA_dataset(data_path, dataset, best_step)
    model = load_model(os.path.join(model_path, 'rl_model'), dataset)

    interval = int(test.shape[2] / 10)
    nsteps = [i * interval for i in range(1, 11)]
    print(nsteps)

    print(Counter(train_labels))
    n_samples = Counter(train_labels)[0]
    print(train_labels[np.where(train_labels == 0)])

    print('=' * 50)
    for nstep in nsteps:
        new_train, new_test = train[:, :, :nstep], test[:, :, :nstep]
        new_train, new_test = preprocess(new_train, new_test, test.shape[1])

        train_features = model.encode(new_train, batch_size=50)
        test_features = model.encode(new_test, batch_size=50)
        print(new_train.shape, train_features.shape, new_test.shape, test_features.shape)

        cluster_model = cluster(train_features)
        cluster_model_path = os.path.join(model_path, 'cluster_model', dataset, dataset + '_' + str(nstep) + '.pkl')
        joblib.dump(cluster_model, cluster_model_path)
        # cluster_model = joblib.load(cluster_model_path)

        y_test_pred, test_prob = assign_cluster(cluster_model, test_features)

        print(y_test_pred[np.where(test_labels == 0)])
        print(Counter(y_test_pred))

        print('='*50)
