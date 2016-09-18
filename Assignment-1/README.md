# Assignment 1 - Recommender Systems #

## Usage ##
```sh
usage: evaluate_recommender.py [-h] [-e {rmse,mae}]
          {naive-global,naive-user,naive-item,naive-regression,matrix-factorization}
```

## Results ##
| Algorithm            	|        RMSE 	|        MAE   	|
|----------------------	|-------	|-------	|-------	|-------	|
|                      	| Train 	|  Test 	| Train 	|  Test 	|
| Naive Global         	| 1.117 	| 1.117 	| 0.934 	| 0.934 	|
| Naive User           	| 1.028 	| 1.036 	| 0.823 	| 0.829 	|
| Naive Item           	| 0.974 	| 0.980 	| 0.778 	| 0.782 	|
| Naive Regression     	| 0.915 	| 0.924 	| 0.725 	| 0.733 	|
| Matrix Factorization 	|       	|       	|       	|       	|
