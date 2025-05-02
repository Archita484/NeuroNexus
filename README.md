# NeuroNexus
Titanic Survival Prediction using Machine Learning

This project uses the famous Titanic dataset from Kaggle to predict whether a passenger survived or not, based on features such as age, gender, ticket class, and more.


Table of Contents
	•	Overview
	•	Dataset
	•	Libraries Used
	•	Data Preprocessing
	•	Model Building
	•	Evaluation
	•	Results
	•	Future Improvements


Overview

The goal of this machine learning project is to predict the survival of passengers on the Titanic using supervised learning algorithms. This is a classic binary classification problem.


Dataset
	•	The dataset is from Kaggle’s Titanic: Machine Learning from Disaster competition.

Files used:
	•	train.csv: Used to train and validate the model.



Libraries Used
	•	pandas — for data manipulation
	•	numpy — for numerical operations
	•	matplotlib & seaborn — for data visualization
	•	sklearn — for machine learning modeling and evaluation



Data Preprocessing
	•	Missing values in Age were filled using the median age.
	•	Missing values in Embarked were filled using the mode.
	•	Dropped less useful columns: Cabin, Ticket, Name, PassengerId
	•	Converted categorical data (Sex, Embarked) into numerical values using Label Encoding.


Model Building
	•	Used Random Forest Classifier from scikit-learn.
	•	Data was split into 80% training and 20% testing using train_test_split.


Evaluation
	•	Evaluated using accuracy score and classification report (precision, recall, f1-score).
	•	Printed the accuracy on test data and interpreted performance.


Results
	•	The model achieved an accuracy of approximately ~80% on the test set.
	•	Gender and Passenger Class were the most important features in predicting survival.


 Future Improvements
	•	Use other models like Logistic Regression, KNN, or SVM for comparison.
	•	Tune hyperparameters using GridSearchCV.
	•	Use feature engineering (like creating FamilySize from SibSp and Parch).
	•	Visualize predictions using confusion matrix.
