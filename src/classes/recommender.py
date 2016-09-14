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

	def get_error_estimation(self, train_set, test_set, estimator, prediction):
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
		# prediction = self.get_prediction(train_set)
		self.estimator = estimator
		if self.recommender == 'naive-global':
			train = self.get_estimate(train_set[:, 2], prediction)
			test = self.get_estimate(test_set[:, 2], prediction)
		elif self.recommender == 'naive-user':
			train = self.get_estimate(train_set[:, [0, 2]], prediction)
			test = self.get_estimate(test_set[:, [0, 2]], prediction)
		elif self.recommender == 'naive-item':
			train = self.get_estimate(train_set[:, [1, 2]], prediction)
			test = self.get_estimate(test_set[:, [1, 2]], prediction)
		return train, test

	def get_prediction(self, array, size=None):
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
			return self.naive_user(array, size), self.naive_global(array)
		elif self.recommender == 'naive-item':
			return self.naive_item(array, size), self.naive_global(array)

	def naive_global(self, array):
		"""
		This function computes the Global Average Score Recommender.

		Args:
			array: An array with the values whose average score we compute

		Returns:
			The average score of the input

		"""
		return np.mean(array[:, 2])

	def naive_user(self, array, total):
		"""
		This function returns the predictions for naive user recommender

		Args:
			array: An array with the values of the type and the rating(2 dimensional)
			total: The total amount of users in the data
		Returns:
			An array with the average score for each user. The size of the array is
			total + 1, since we do not have user 0. Items that are missing
			from the array get nan.
		"""
		return self.array_average(array[:, [0, 2]], total)

	def naive_item(self, array, total):
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
		return self.array_average(array[:, [1, 2]], total)

	def array_average(self, array, total):
		"""
		This function computes the [Type] Average Score Recommender(User or Item).

		Args:
			array: An array with the values of the type and the rating(2 dimensional)
			total: The total amount of the items of the type(either movie or user)
					in the data
		Returns:
			An array with the average score for each item. The size of the array is
			total + 1, since we do not have user 0 or movie 0. Items that are missing
			from the array get nan.
		"""
		# Create an array to hold the average
		averages = np.zeros(array[:, 0].max(axis=0) + 1)
		# Count the occurences of each item
		occur = np.bincount(array[:, 0])
		# Sum ratings for each item in the array
		for entry in array:
			averages[entry[0]] += entry[1]
		# Get the average
		result = np.divide(averages, occur)
		# Append missing values in array from the full dataset
		result = np.append(
			result,
			np.ones(total - len(result) + 1) * np.nan)
		return result
