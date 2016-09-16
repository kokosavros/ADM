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
			prediction: The prediction of the model

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
		elif self.recommender == 'naive-regression':
			train = self.get_estimate(train_set, prediction)
			test = self.get_estimate(test_set, prediction)
		return train, test

	def get_prediction(self, array, size=None):
		"""
		Get the prediction based on the recommender algorithm selected

		Args:
			array: The set with the values on which we compute the prediction
			size: Optional argument to pass size of user or items
		Returns:
			The prediction in float number
		"""
		global_average = self.naive_global(array)
		if self.recommender == 'naive-global':
			return global_average
		elif self.recommender == 'naive-user':
			prediction = self.naive_user(array, size[0])
			prediction[np.isnan(prediction)] = global_average
			return prediction
		elif self.recommender == 'naive-item':
			prediction = self.naive_item(array, size[1])
			prediction[np.isnan(prediction)] = global_average
			return prediction
		elif self.recommender == 'naive-regression':

			r_users_items = array[:, 2]

			prediction_items = self.naive_item(array, size[1])
			prediction_users = self.naive_user(array, size[0])
			r_items = np.zeros(len(r_users_items))
			r_users = np.zeros(len(r_users_items))

			index = 0
			for item in array:
				r_items[index] = prediction_items[item[1]]
				r_users[index] = prediction_users[item[0]]
				index += 1

			A = np.vstack([r_users, r_items, np.ones(len(r_users_items))]).T
			a, b, c = np.linalg.lstsq(A, r_users_items)[0]
			prediction = np.full((size[0] + 1, size[1] + 1), np.nan)
			for x in range(size[0] + 1):
				for y in range(size[1] + 1):
					prediction[x, y] = \
						a * prediction_users[x] +\
						b * prediction_items[y] +\
						c
			# print prediction
			prediction[np.isnan(prediction)] = global_average
			return prediction

	def naive_global(self, array):
		"""
		This function computes the Global Average Score Recommender.

		Args:
			array: An array with the values whose average score we compute

		Returns:
			The average score of the input

		"""
		return np.mean(array[:, 2])

	def naive_user(self, array, total_users):
		"""
		This function returns the predictions for naive user recommender

		Args:
			array: An array with the values of the type and the rating(2 dimensional)
			total_users: The total amount of users in the data
		Returns:
			An array with the average score for each user. The size of the array is
			total + 1, since we do not have user 0. Items that are missing
			from the array get nan.
		"""
		return self.array_average(array[:, [0, 2]], total_users)

	def naive_item(self, array, total_items):
		"""
		This function returns the predictions for naive user recommender

		Args:
			array: An array with the values of the type and the rating(2 dimensional)
			total_items: The total amount of items in the data
		Returns:
			An array with the average score for each item. The size of the array is
			total + 1, since we do not movie 0. Items that are missing
			from the array get nan.
		"""
		return self.array_average(array[:, [1, 2]], total_items)

	@staticmethod
	def array_average(array, total):
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

	@staticmethod
	def get_utility_matrix(array, size, prediction=None):
		"""
		Return an ndarray to serve as a utillity matrix. If we have
		a prediction from a model, use these predictions for the values
		we don't know

		Args:
			array: THe array with the initial values we have. It can be
				a train set.
			size: The size of our complete dataset
			prediction: The prediction we have from naive-user or naive item
				algorithms

		Returns:
			A numpy ndarray filled with values and predictions. Its size is
			size + 1 in both directions.

		"""		
		if type(prediction) == np.float64:
			matrix = np.full((size[0] + 1, size[1] + 1), prediction)
			for rating in array:
				matrix[rating[0], rating[1]] = rating[2]
			return matrix
		matrix = np.full((size[0] + 1, size[1] + 1), np.nan)
		if len(prediction) == size[1] + 1:
			index = 0
			for rating in prediction:
				for row in matrix:
					row[index] = rating
				index += 1
			for rating in array:
				matrix[rating[0], rating[1]] = rating[2]
			return matrix
		index = 0
		for rating in prediction:
			matrix[index] = rating
			index += 1
		for rating in array:
			matrix[rating[0], rating[1]] = rating[2]

		return matrix
