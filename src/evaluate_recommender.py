import numpy as np
import argparse
import os
from classes.estimator import Estimator
from classes.recommender import Recommender
import export

# Parse the arguments
alg_choices = ['naive-global', 'naive-user', 'naive-item', 'naive-regression']
est_choices = ['rmse', 'mae']

parser = argparse.ArgumentParser()
# Add algorithm argument
parser.add_argument(
    "algorithm",
    help="The algorithm you want to run",
    choices=alg_choices
)
# Add estimator argument
parser.add_argument(
    '-e', '--estimator',
    help='The error estimator',
    default=None,
    choices=est_choices
)

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
dataset_size = [users, movies]

# Split data into 5 train and test folds
folds = 5

# To make sure we are able to repeat results, set the random seed to something:
np.random.seed(17)

# Make an array of size == len(ratings) with values from 0-5
seqs = [x % folds for x in range(len(ratings))]
# Randomize its order to put entries into folds
np.random.shuffle(seqs)

# Array to save the errors
errors = np.zeros((2, len(est_choices)), dtype=np.float64)

# For each fold:
for fold in range(folds):
    # Build train sets and test set
    train = ratings[np.array([x != fold for x in seqs])]
    test = ratings[np.array([x == fold for x in seqs])]

    # Create a recommender object
    recommender = Recommender(algorithm, train, test, dataset_size)

    # Get prediction
    prediction = recommender.get_prediction()

    # Compute the sum of the errors
    errors = np.add(
        errors,
        recommender.get_error_estimation(estimator, prediction)
    )

filename = '../results/' + algorithm + '.txt'

export.save_results(filename, errors / folds)

print('Results were saved in \'../results/%s.txt\'' % algorithm)
