import unittest
import numpy as np
import recommender


class TestRecommenderMethods(unittest.TestCase):

    def test_naive_type(self):
        global_avg = 1
        array = np.array([[2, 4], [1, 5], [67, 2], [23, 1], [23, 4]])
        total_entries = 100
        test = recommender.Recommender(None)
        result = test.naive_type(array, total_entries)
        self.assertEqual(len(result), total_entries + 1)
        self.assertEqual(result[23], 2.5)
        self.assertEqual(result[2], 4)
        self.assertEqual(np.isnan(result[3]), True)
        self.assertEqual(np.isnan(result[0]), True)
        self.assertEqual(np.isnan(result[100]), True)


if __name__ == '__main__':
    unittest.main()
