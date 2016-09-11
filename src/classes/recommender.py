import numpy as np
import estimator
import warnings


class Recommender(estimator.Estimator):
	"""
	The various recommender algorithms to be tested.
	It inherits from the estimator class to be able to
	directly get error estimations.
	"""
	def __init__(self, recommender):

		#warnings.simplefilter("error")
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
		if self.recommender == 'naive-global':
			train = self.get_estimate(train_set, prediction)
			test = self.get_estimate(test_set, prediction)
		elif self.recommender == 'naive-user':
			train = self.get_estimate(train_set, prediction)
			test = self.get_estimate(test_set, prediction)
		elif self.recommender == 'naive-item':
			prediction = prediction[np.newaxis]
			train = self.get_estimate(train_set, prediction.T)
			test = self.get_estimate(test_set, prediction.T)	
		return train, test

	def get_prediction(self, array):
		"""
		Get the prediction based on the recommender algorithm selected

		Args:
			array: The set with the values on which we comput the prediction
			size: Optional argument to pass size of user or items
		Returns:
			The prediction in float number
		"""
		if self.recommender == 'naive-global':
			return self.naive_global(array)
		elif self.recommender == 'naive-user':
			return self.naive_user(array)
		elif self.recommender == 'naive-item':
			return self.naive_item(array)

	def naive_global(self, array):
		"""
		This function computes the Global Average Score Recommender.

		Args:
			array: An array with the values whose average score we compute

		Returns:
			The average score of the input

		"""
		return np.nanmean(array)

	def naive_user(self, array):
		"""
		This function returns the predictions for naive user recommender

		Args:
			array: An array with the values of the type and the rating(2 dimensional)
		Returns:
			An array with the average score for each user.
		"""
		return np.nanmean(array, axis=0)

	def naive_item(self, array):
		"""
		This function returns the predictions for naive user recommender

		Args:
			array: An array with the values of the type and the rating(2 dimensional)
			total: The total amount of items in the data
		Returns:
			An array with the average score for each item. The size of the array is
			total + 1, since we do not movie 0. Items that are missing
			from the array get nan.
		"""
		return np.nanmean(array, axis=1)
