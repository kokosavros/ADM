import numpy as np
import argparse
import os
from classes.estimator import Estimator
from classes.recommender import Recommender


# Parse the arguments
choices = ['naive-global']
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
np.random.seed(17)

# Make an array of size == len(ratings) with values from 0-5
seqs = [x % folds for x in range(len(ratings))]
# Randomize its order to put entries into folds
np.random.shuffle(seqs)

# For each fold:
for fold in range(folds):
    train_sel = np.array([x != fold for x in seqs])
    test_sel = np.array([x == fold for x in seqs])
    train = ratings[train_sel]
    test = ratings[test_sel]

    # Calculate model parameters: mean rating over the training set:
    recommender = Recommender(algorithm)
    global_average = recommender.get_prediction(train[:, 2])

    '''
    movie_avg = averages.get_array_averages(train[:, [1, 2]], movies, global_average)
    user_avg = averages.get_array_averages(train[:, [0, 2]], users, global_average)
    user_avg[np.isnan(user_avg)] = global_average
    movie_avg[np.isnan(movie_avg)] = global_average
    '''

    if estimator == 'all':
        index = 0
        for err_estimator in estimator_choices:
            errors = recommender.get_error_estimation(
                train[:, 2], test[:, 2], err_estimator)

            err_train[index, fold] = errors[0]
            err_test[index, fold] = errors[1]
            index += 1
    else:
        errors = recommender.get_error_estimation(
            train[:, 2], test[:, 2], estimator)
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
