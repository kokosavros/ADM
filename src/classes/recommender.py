import numpy as np
import estimator


class Recommender(estimator.Estimator):
	"""
	The various recommender algorithms to be tested.
	It inherits from the estimator class to be able to
	directly get error estimations.
	"""
	def __init__(self, recommender):
		self.recommender = recommender

	def get_error_estimation(self, train_set, test_set, estimator):
		"""
		Get the errors for on the train set and the test set, based
		on the desired estimator.

		Args:
			train_set: The training set
			test_set: The test set
			estimator: The type of the estimator function(RMSE, MAE, etc)

		Returns:
			An array with the computed errors on the train set and the test
			set
		"""
		prediction = self.get_prediction(train_set)
		self.estimator = estimator
		train = self.get_estimate(train_set, prediction)
		test = self.get_estimate(test_set, prediction)
		return train, test

	def get_prediction(self, values):
		"""
		Get the prediction based on the recommender algorithm selected

		Args:
			values: The set with the values on which we comput the prediction

		Returns:
			The prediction in float number
		"""
		if self.recommender == 'naive-global':
			return self.naive_global(values)

	def naive_global(self, values):
		"""
		This function computes the Global Average Score Recommender.

		Args:
			values: An array with the values whose average score we compute

		Returns:
			The average score of the input

		"""
		return np.mean(values)
