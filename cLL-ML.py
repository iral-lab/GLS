import argparse
from datetime import datetime
import os
import pickle
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset import DataSet

np.set_printoptions(threshold=np.inf, suppress=True, precision=3)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', default='results',
                        help='File path to log output')
    parser.add_argument('--feature_types', nargs='+', choices=['rgb', 'shape', 'object'], default=['rgb', 'shape', 'object'],
                        help='feature type to learn')
    parser.add_argument('--annotation_file', default='conf_files/UW_english/UW_AMT_description_documents_per_image_nopreproc_stop_raw.conf',
                        help='Amazon Mechanical Turk annotation file')
    parser.add_argument('--cutoff', choices=[0.25, 0.5, 0.75], default=0.25, type=float,
                        help='The cosine similarity cutoff for negative example generation')
    parser.add_argument('--feature_directory', default="../ImgDz",
                        help='folder for features')
    parser.add_argument('--dataset_file', default='dataset.pkl',
                        help='Filename of a DataSet object for convenience')
    parser.add_argument('--test_percent', type=float, default=0.2,
                        help='Percentage of instances to set aside for testing')

    return parser.parse_known_args()

def get_features_from_file(instance, feature_type, feature_directory):
    """
    Read feature data from file and return as flat list

    :param string instance: "water_bottle/water_bottle_1_4_55"
    :param string feature_type: type of feature (rgb, shape, or object)
    :param string feature_directory: path where features are stored
    :return: features read from file in a flat lists
    """
    category, instance = instance.split("/")
    # TODO: could/should probably do this with os.path library
    path = f'{feature_directory}/{category}/{instance}/{instance}_{feature_type}.log'

    features = []
    with open(path, 'r') as fin:
        for line in fin:
            features += [float(datum) for datum in line.strip().split(',')]

    return features

def get_feature_data(dataset, test_instances, feature_types, feature_directory):
    # This is the variable that defines how many positive instances a token must
    # have before it is deemed useful.
    # NOTE: this is different from how many times it appeared for a particular
    # instance. For the training that appears to be 1.
    MIN_POSITIVE_INSTANCES = 3
    MIN_NEGATIVE_INSTANCES = 2

    for token in dataset.tokens:
        # Avoid key errors and tokens with no positive or negative instances
        if (token not in dataset.pos_tokens_to_instances
                or token not in dataset.neg_tokens_to_instances):
            continue

        # Skip tokens who have few positive instances associated with them
        if (len(dataset.pos_tokens_to_instances[token]) < MIN_POSITIVE_INSTANCES
                or len(dataset.neg_tokens_to_instances[token]) < MIN_NEGATIVE_INSTANCES):
            continue

        for feature_type in feature_types:
            pos_features = []
            for instance in dataset.pos_tokens_to_instances[token]:
                if instance in test_instances:
                    continue
                pos_features.append(get_features_from_file(instance, feature_type, feature_directory))

            neg_features = []
            for instance in dataset.neg_tokens_to_instances[token]:
                if instance in test_instances:
                    continue
                neg_features.append(get_features_from_file(instance, feature_type, feature_directory))

            # balance the number of features
            # don't need to check for division by zero since length was already checked
            if len(pos_features) > len(neg_features):
                neg_features *= len(pos_features) // len(neg_features)

            if len(neg_features) > len(pos_features):
                pos_features *= len(neg_features) // len(pos_features)

            X = pos_features + neg_features
            y = ([1] * len(pos_features)) + ([0] * len(neg_features))

            yield (X, y, feature_type, token)

def main():
    ARGS, unused = parse_args()
    # TODO: should be a cmd line arg
    TEST_PERCENTAGE = 0.2

    if not os.path.exists(ARGS.feature_directory):
        print("Feature directory does not exist")
        return

    if not os.path.exists(ARGS.output_directory):
        os.mkdir(ARGS.output_directory)

    for feature_type in ARGS.feature_types:
        if not os.path.exists(ARGS.output_directory + '/' + feature_type):
            os.mkdir(ARGS.output_directory + '/' + feature_type)

    print("START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists(ARGS.dataset_file):
        print('Found dataset pickle file, loading saved dataset...')
        with open(ARGS.dataset_file, 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        print(f'Couldn\'t find {ARGS.dataset_file}, getting new dataset...')
        dataset = DataSet(ARGS.annotation_file)
        print('Writing DataSet to pickle file')
        with open(ARGS.dataset_file, 'wb') as fout:
            pickle.dump(dataset, fout)

    # these instances should not be used during training
    test_instances = random.sample(dataset.data.keys(), int(len(dataset.data) * ARGS.test_percent))

    # Store test_data in memory for testing without grabbing feature data from file every time
    print('Preloading test data...')
    test_data = {'rgb': {}, 'shape': {}, 'object': {}}
    for feature_type in ARGS.feature_types:
        for instance in test_instances:
            test_data[feature_type][instance] = get_features_from_file(instance, feature_type, ARGS.feature_directory)

    print("ML START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Train and run binary classifiers for all tokens, find the probabilities
    # for the associations between all tokens and test instances,
    # and log the probabilities

    for X, y, feature_type, token in get_feature_data(dataset, test_instances, ARGS.feature_types, ARGS.feature_directory):
        print(f'Training model on token \"{token}\"...')
        model = LogisticRegression(solver='lbfgs', max_iter=500)
        model.fit(X, y)

        print(f'Testing model and writing output...')
        output_file = ARGS.output_directory + f'/{feature_type}/{token}.txt'
        with open(output_file, 'w') as fout:
            fout.write('test_instance,y,predicted\n')
            for instance, X_test in test_data[feature_type].items():
                y_test = 1 if instance in dataset.pos_tokens_to_instances[token] else 0
                probabilities = model.predict_proba([X_test])
                fout.write(f'{instance},{y_test},{probabilities[0]}\n')

    print("ML END :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
