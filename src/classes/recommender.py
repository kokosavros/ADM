import numpy as np
import estimator


class Recommender(estimator.Estimator):
	"""
	The various recommender algorithms to be tested.
	It inherits from the estimator class to be able to
	directly get error estimations.
	"""
	def __init__(self, recommender, train_set, test_set, size):
		self.recommender = recommender
		self.train_set = train_set
		self.test_set = test_set
		self.size = size

	def get_error_estimation(self, estimator, prediction):
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
			train = self.get_estimate(self.train_set[:, 2], prediction)
			test = self.get_estimate(self.test_set[:, 2], prediction)
		elif self.recommender == 'naive-user':
			train = self.get_estimate(self.train_set[:, [0, 2]], prediction)
			test = self.get_estimate(self.test_set[:, [0, 2]], prediction)
		elif self.recommender == 'naive-item':
			train = self.get_estimate(self.train_set[:, [1, 2]], prediction)
			test = self.get_estimate(self.test_set[:, [1, 2]], prediction)
		elif self.recommender == 'naive-regression':
			train = self.get_estimate(self.train_set, prediction)
			test = self.get_estimate(self.test_set, prediction)
		return train, test

	def get_prediction(self):
		"""
		Get the prediction based on the recommender algorithm selected

		Returns:
			The prediction in float number
		"""
		global_average = self.naive_global(self.train_set)
		if self.recommender == 'naive-global':
			return global_average
		elif self.recommender == 'naive-user':
			prediction = self.naive_user(self.train_set, self.size[0])
			prediction[np.isnan(prediction)] = global_average
			return prediction
		elif self.recommender == 'naive-item':
			prediction = self.naive_item(self.train_set, self.size[1])
			prediction[np.isnan(prediction)] = global_average
			return prediction
		elif self.recommender == 'naive-regression':
			prediction_items = self.naive_item(self.train_set, self.size[1])
			prediction_users = self.naive_user(self.train_set, self.size[0])

			r_users_items = self.train_set[:, 2]
			r_items = prediction_items[self.train_set[:, 1]]
			r_users = prediction_users[self.train_set[:, 0]]

			A = np.vstack(
				[
					r_users,
					r_items,
					np.ones(len(r_users_items))
				]
			).T

			a, b, c = np.linalg.lstsq(A, r_users_items)[0]
			prediction = np.full((self.size[0] + 1, self.size[1] + 1), np.nan)

			for x in range(self.size[0] + 1):
				for y in range(self.size[1] + 1):
					prediction[x, y] = \
						a * prediction_users[x] +\
						b * prediction_items[y] +\
						c

			prediction[np.isnan(prediction)] = \
				(np.nanmean(prediction_users) + np.nanmean(prediction_items)) / 2
			prediction = np.clip(prediction, 1, 5)
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
