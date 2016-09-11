import numpy as np



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
		return np.sqrt(np.mean((values - pred_value)**2))

	def mae(self, values, pred_value):
		"""
		The Mean Absolute Error

		Args:
			values: The set on which we compute the error estimation
			pred_value: The value predicted by the model we have used

		Returns:
			The error between the real values and the predicted values
		"""
		return np.nanmean(np.absolute((values - pred_value)))
