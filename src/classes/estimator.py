import numpy as np
import math


class Estimator():
	"""
	The class containing the different estimators we use.
	"""
	def __init__(self, estimator):
		self.estimator = estimator

	def get_estimate(self, values, pred_value):
		"""
		Get the estimate based on the estimator set.

		Args:
			values: The set on which we compute the error estimation
			pred_value: The value predicted by the model we have used

		Returns:
			The error between the real values and the predicted values
		"""
		if self.estimator == 'rmse':
			return self.rmse(values, pred_value)
		return self.mae(values, pred_value)

	def rmse(self, values, pred_value):
		"""
		The Root Mean Squared Error

		Args:
			values: The set on which we compute the error estimation
			pred_value: The value predicted by the model we have used

		Returns:
			The error between the real values and the predicted values
		"""
		if type(pred_value) == np.float64:
			return np.sqrt(np.mean((values - pred_value)**2))
		error = 0
		index = 0
		for row in values:
			error += (row[1] - pred_value[row[0]])**2
			index += 1
		return math.sqrt(error / index)

	def mae(self, values, pred_value):
		"""
		The Mean Absolute Error

		Args:
			values: The set on which we compute the error estimation
			pred_value: The value predicted by the model we have used

		Returns:
			The error between the real values and the predicted values
		"""
		if type(pred_value) == np.float64:
			return np.mean(np.absolute((values - pred_value)))
		error = 0
		index = 0
		for row in values:
			error += math.fabs(row[1] - pred_value[row[0]])
			index += 1
		return error / index
