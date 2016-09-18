# Assignment 1 - Recommender Systems #

## Assignment Structure ##
```
README.md
src/
	+-- evaluate_recommender.py
	+-- export.py
	+-- datasets/
		+-- README
		+-- movies.dat
		+-- ratings.dat
		+-- users.dat
	+-- classes/
		+-- recommender.py
		+-- estimator.py
		+-- test_recommender.py
```
## General Info ##

#### export.py ####
File with the function to export data

#### evaluate_recommender.py ####
The main file of the assignment. It uses 5-fold cross validation to evaluate the implemented recommenders.

#### recommender.py ####
The recommender class. In initialization you define the recommender algorithm and you pass the training and the test set, as well as the size of the initial dataset. 

You can get a prediction for each model, as well as estimation of the errors for the models.

#### estimator.py ####
The estimator class. It offers estimations for RMSE and MAE errors.

## Usage ##
```sh
usage: evaluate_recommender.py [-h] [-e {rmse,mae}]
          {naive-global,naive-user,naive-item,naive-regression,matrix-factorization}
```

## Results ##
| Algorithm            	|  RMSE 	|       	| MAE   	|       	|
|----------------------	|:-----:	|-------	|-------	|-------	|
|                      	| Train 	|  Test 	| Train 	|  Test 	|
| Naive Global         	| 1.117 	| 1.117 	| 0.934 	| 0.934 	|
| Naive User           	| 1.028 	| 1.036 	| 0.823 	| 0.829 	|
| Naive Item           	| 0.974 	| 0.980 	| 0.778 	| 0.782 	|
| Naive Regression     	| 0.915 	| 0.924 	| 0.725 	| 0.733 	|
| Matrix Factorization 	|       	|       	|       	|       	|

## Times ##
| Algorithm            	| Time Per Call 	|
|----------------------	|:-------------:	|
| Naive Global         	|     0.002     	|
| Naive User           	|     0.839     	|
| Naive Item           	|     0.811     	|
| Naive Regression     	|     32.394    	|
| Matrix Factorization 	|               	|
