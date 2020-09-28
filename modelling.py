import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from feature_engineering import FeatureEngineering
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

with open("features_train.pickle", 'rb') as output :
	features_train = pickle.load(output)

with open("labels_train.pickle", 'rb') as output :
	labels_train = pickle.load(output)

with open("features_test.pickle", 'rb') as output :
	features_test = pickle.load(output)

with open("labels_test.pickle", 'rb') as output :
	labels_test = pickle.load(output)

class modellig :

	# Build base model using Random Forest Classifier
	# def RFC(self,features_train, labels_train,features_test, labels_test,clean_data_frame):
		# print(features_train.shape)
		# print(features_test.shape)

		# rf_0 = RandomForestClassifier(random_state = 8)
		# print('Parameters currently in use:\n')
		# print(rf_0.get_params())

		# # n_estimators
		# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

		# # max_features
		# max_features = ['auto', 'sqrt']

		# # max_depth
		# max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
		# max_depth.append(None)

		# # min_samples_split
		# min_samples_split = [2, 5, 10]

		# # min_samples_leaf
		# min_samples_leaf = [1, 2, 4]

		# # bootstrap
		# bootstrap = [True, False]

		# # Create the random grid
		# random_grid = {'n_estimators': n_estimators,
		#                'max_features': max_features,
		#                'max_depth': max_depth,
		#                'min_samples_split': min_samples_split,
		#                'min_samples_leaf': min_samples_leaf,
		#                'bootstrap': bootstrap}

		# print(random_grid)

		# # First create the base model to tune
		# rfc = RandomForestClassifier(random_state=8)

		# # Definition of the random search
		# random_search = RandomizedSearchCV(estimator=rfc,
		#                                    param_distributions=random_grid,
		#                                    n_iter=50,
		#                                    scoring='accuracy',
		#                                    cv=3, 
		#                                    verbose=1, 
		#                                    random_state=8)

		# # Fit the random search model
		# random_search.fit(features_train, labels_train)

		# print("The best hyperparameters from Random Search are:")
		# print(random_search.best_params_)
		# print("")
		# print("The mean accuracy of a model with these hyperparameters is:")
		# print(random_search.best_score_)

		# Create the parameter grid based on the results of random search
		# bootstrap = [False]
		# max_depth = [30, 40, 50]
		# max_features = ['sqrt']
		# min_samples_leaf = [1, 2, 4]
		# min_samples_split = [5, 10, 15]
		# n_estimators = [800]

		# param_grid = {
		#     'bootstrap': bootstrap,
		#     'max_depth': max_depth,
		#     'max_features': max_features,
		#     'min_samples_leaf': min_samples_leaf,
		#     'min_samples_split': min_samples_split,
		#     'n_estimators': n_estimators
		# }

		# # Create a base model
		# rfc = RandomForestClassifier(random_state=8)

		# # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
		# cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

		# # Instantiate the grid search model
		# grid_search = GridSearchCV(estimator=rfc, 
		#                            param_grid=param_grid,
		#                            scoring='accuracy',
		#                            cv=cv_sets,
		#                            verbose=1)

		# # Fit the grid search to the data
		# grid_search.fit(features_train, labels_train)

		# print("The best hyperparameters from Grid Search are:")
		# print(grid_search.best_params_)
		# print("")
		# print("The mean accuracy of a model with these hyperparameters is:")
		# print(grid_search.best_score_)


		# best_rfc = grid_search.best_estimator_
		# best_rfc

		# best_rfc.fit(features_train, labels_train)
		# rfc_pred = best_rfc.predict(features_test)

		# # Training accuracy
		# print("The training accuracy is: ")
		# print(accuracy_score(labels_train, best_rfc.predict(features_train)))

		# # Test accuracy
		# print("The test accuracy is: ")
		# print(accuracy_score(labels_test, rfc_pred))


		# # Classification report
		# print("Classification report")
		# print(classification_report(labels_test,rfc_pred))

		# aux_df = clean_data_frame[['Sentiment Class', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
		# conf_matrix = confusion_matrix(labels_test, rfc_pred)
		# plt.figure(figsize=(12.8,6))
		# sns.heatmap(conf_matrix, 
		#             annot=True,
		#             xticklabels=aux_df['Sentiment Class'].values, 
		#             yticklabels=aux_df['Sentiment Class'].values,
		#             cmap="Blues")
		# plt.ylabel('Predicted')
		# plt.xlabel('Actual')
		# plt.title('Confusion matrix')
		# plt.show()

		# base_model = RandomForestClassifier(random_state = 8)
		# base_model.fit(features_train, labels_train)
		# accuracy_score(labels_test, base_model.predict(features_test))

		# best_rfc.fit(features_train, labels_train)
		# accuracy_score(labels_test, best_rfc.predict(features_test))

		# d = {
		#      'Model': 'Random Forest',
		#      'Training Set Accuracy': accuracy_score(labels_train, best_rfc.predict(features_train)),
		#      'Test Set Accuracy': accuracy_score(labels_test, rfc_pred)
		# }

		# df_models_rfc = pd.DataFrame(d, index=[0])

		# df_models_rfc

		# with open('Models/best_rfc.pickle', 'wb') as output:
		#     pickle.dump(best_rfc, output)
		
		# with open('Models/df_models_rfc.pickle', 'wb') as output:
		#     pickle.dump(df_models_rfc, output)

		# using multinomial naive bayes classifier

	def MNB(self,features_train, labels_train,features_test, labels_test,clean_data_frame):

		print(features_train.shape)
		print(features_test.shape)

		mnbc = MultinomialNB()
		mnbc

		mnbc.fit(features_train, labels_train)
		mnbc_pred = mnbc.predict(features_test)
		# Training accuracy
		print("The training accuracy is: ")
		print(accuracy_score(labels_train, mnbc.predict(features_train)))

		# Test accuracy
		print("The test accuracy is: ")
		print(accuracy_score(labels_test, mnbc_pred))

		# Classification report
		print("Classification report")
		print(classification_report(labels_test,mnbc_pred))

		aux_df = clean_data_frame[['Sentiment Class', 'category_code']].drop_duplicates().sort_values('category_code')
		conf_matrix = confusion_matrix(labels_test, mnbc_pred)
		plt.figure(figsize=(12.8,6))
		sns.heatmap(conf_matrix/np.sum(conf_matrix),
		            annot=True,
		            fmt='.2%',
		            xticklabels=aux_df['Sentiment Class'].values, 
		            yticklabels=aux_df['Sentiment Class'].values,
		            cmap="Blues")
		plt.ylabel('Predicted')
		plt.xlabel('Actual')
		plt.title('Confusion matrix')
		plt.show()

		d = {
		     'Model': 'Multinomial Naïve Bayes',
		     'Training Set Accuracy': accuracy_score(labels_train, mnbc.predict(features_train)),
		     'Test Set Accuracy': accuracy_score(labels_test, mnbc_pred)
		}

		df_models_mnbc = pd.DataFrame(d, index=[0])

		df_models_mnbc

		with open('df_models_mnbc.pickle', 'wb') as output:
			pickle.dump(df_models_mnbc, output)

	def main(self):
		FE=FeatureEngineering()
		features_train, labels_train,features_test, labels_test,clean_data_frame=FE.main()
		# RandomforestC=self.RFC(features_train, labels_train,features_test, labels_test,clean_data_frame)
		MultinomialNB = self.MNB(features_train, labels_train,features_test, labels_test,clean_data_frame)

if __name__ == '__main__':
	KNN = modellig()
	KNN.main()


