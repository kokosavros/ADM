import numpy as np


def save_results(filename, results):
	with open(filename, 'w') as output:
		if np.isfinite(results[0, 0]):
			output.write('RMSE\n')
			output.write("Mean error on TRAIN: %s\n" % results[0, 0])
			output.write("Mean error on  TEST: %s\n" % results[1, 0])
		if np.isfinite(results[0, 1]):
			output.write('MAE\n')
			output.write("Mean error on TRAIN: %s\n" % results[0, 1])
			output.write("Mean error on  TEST: %s\n" % results[1, 1])
