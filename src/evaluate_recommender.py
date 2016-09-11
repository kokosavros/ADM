import numpy as np
import numpy.ma as ma
import argparse
import os
from classes.estimator import Estimator
from classes.recommender import Recommender


# Parse the arguments
choices = ['naive-global', 'naive-user', 'naive-item']
estimator_choices = ['rmse', 'mae']

parser = argparse.ArgumentParser()
parser.add_argument(
    "algorithm",
    help="The algorithm you want to run",
    choices=choices)
parser.add_argument(
    '-e', '--estimator', help='The error estimator',
    default='all', choices=['rmse', 'mae', 'all'])
args = parser.parse_args()

algorithm = args.algorithm
estimator = args.estimator

print('Evaluating %s algorithm with %s estimator.' % (algorithm, estimator))
# Find if results directory exists
if not os.path.isdir('../results'):
    # Create it
    os.mkdir('../results')

# Open ratings.dat and put it in a numpy array
ratings = np.genfromtxt(
    "../datasets/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

users, movies, rat = ratings.max(axis=0)
# Create utility matrix
utility = np.full((users + 1, movies + 1), np.nan)
for rating in ratings:
    utility[rating[0], rating[1]] = rating[2]

# print utility

# Split data into 5 train and test folds
folds = 5

# Allocate memory for results:
# 2 dimensional to keep both RMSE and MAE
if estimator == 'all':
    err_train = np.zeros((len(estimator_choices), folds))
    err_test = np.zeros((len(estimator_choices), folds))
else:
    err_train = np.zeros(folds)
    err_test = np.zeros(folds)

# To make sure we are able to repeat results, set the random seed to something:
np.random.seed(10)

# Make an array of size == len(ratings) with values from 0-5
seqs = [x % folds for x in range((users + 1) * (movies + 1))]
# Randomize its order to put entries into folds
np.random.shuffle(seqs)

seqs = np.reshape(seqs, (users + 1, movies + 1))

# For each fold:
for fold in range(folds):
    train_sel = ma.masked_not_equal(seqs, fold)
    test_sel = ma.masked_equal(seqs, fold)
    train = ma.MaskedArray(utility, mask=train_sel.mask, fill_value=np.nan)
    
    test = ma.MaskedArray(utility, mask=test_sel.mask, fill_value=np.nan)

    # Calculate model parameters: mean rating over the training set:
    recommender = Recommender(algorithm)
    global_average = recommender.get_prediction(train)

    if estimator == 'all':
        index = 0
        for err_estimator in estimator_choices:
            errors = recommender.get_error_estimation(
                train, test, err_estimator)

            err_train[index, fold] = errors[0]
            err_test[index, fold] = errors[1]
            index += 1
    else:
        errors = recommender.get_error_estimation(
            train, test, estimator)
        err_train[fold] = errors[0]
        err_test[fold] = errors[1]

# Output in file
filename = '../results/' + algorithm + '.txt'
with open(filename, 'w') as output:
    output.write('Results for %s:\n' % algorithm)
    if estimator == 'all':
        index = 0
        for err_estimator in estimator_choices:
            output.write(err_estimator.upper() + '\n')
            output.write(
                "Mean error on TRAIN: %s\n" % np.mean(err_train[index, :]))
            output.write(
                "Mean error on  TEST: %s\n" % np.mean(err_test[index, :]))
            index += 1
    else:
        output.write(estimator.upper() + '\n')
        output.write(
            "Mean error on TRAIN: %s\n" % np.mean(err_train))
        output.write(
            "Mean error on  TEST: %s\n" % np.mean(err_test))

print('Results were saved in \'../results/%s.txt\'' % algorithm)
# Just in case you need linear regression: help(np.linalg.lstsq) will tell you
# how to do it!
