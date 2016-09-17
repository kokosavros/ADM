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
		if len(pred_value.shape) == 1:
			errors = np.sum((values[:, 1] - pred_value[values[:, 0]])**2)
			return np.sqrt(errors / len(values))
		errors = np.sum((values[:, 2] - pred_value[values[:, 0], values[:, 1]])**2)
		return np.sqrt(errors / values.shape[0])

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
		if len(pred_value.shape) == 1:
			return np.sum(np.abs(values[:, 1] - pred_value[values[:, 0]])) / len(values)
		return np.sum(
			np.abs(
				values[:, 2] - pred_value[values[:, 0], values[:, 1]]
			)
		) / values.shape[0]
