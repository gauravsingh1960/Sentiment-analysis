import pickle
from data_cleaning import PreProcessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_selection import chi2

with open("clean_data_frame.pickle", 'rb') as output :
	df = pickle.load(output)

class FeatureEngineering:

	def class_normalize(self, clean_data_frame):
		clean_data_frame['category_code'] = clean_data_frame['Sentiment Class'].map({'Negative':1,'Neutral':2,'Positive':0})

		print(clean_data_frame.head())

		return clean_data_frame

	def stopwords_stemming(self,clean_data_frame):
		ps = PorterStemmer()

		stop_words =  stopwords.words('english')
		exclude_words = ["not","no","don't","doesn't", "didn't"]
		stop_words = [word for word in stop_words if word not in set(exclude_words)]
		print(len(stop_words))
		print(stop_words)

		clean_review_list = []
		for review in clean_data_frame["Review_Text"]:
			review = review.split()
			clean_review = []
			for word in review:
				if str(word).lower() not in set(stop_words):
					clean_review.append(ps.stem(word))
			clean_review = " ".join(clean_review)
			clean_review_list.append(clean_review)

		clean_data_frame["Review_Text"] = clean_review_list
		return clean_data_frame

	def class_Lemma(self, clean_data_frame):
		wordnet_lemmatizer = WordNetLemmatizer()
		nrows = len(clean_data_frame)
		lemmatized_text_list = []

		for row in range(0, nrows):
		    
		    # Create an empty list containing lemmatized words
		    lemmatized_list = []
		    
		    # Save the text and its words into an object
		    text = clean_data_frame.loc[row]['Review_Text']
		    text_words = text.split(" ")

		    # Iterate through every word to lemmatize
		    for word in text_words:
		        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

		    # Join the list
		    lemmatized_text = " ".join(lemmatized_list)

		    # Append to the list containing the texts
		    lemmatized_text_list.append(lemmatized_text)

		clean_data_frame['Review_Text'] = lemmatized_text_list
		return clean_data_frame

	def partition (self,clean_data_frame):
		x_train,x_test,y_train,y_test = train_test_split(clean_data_frame["Review_Text"], clean_data_frame["category_code"], test_size = 0.3,random_state=0)
		print(x_train,y_train)

		return x_train,x_test,y_train,y_test, clean_data_frame

	def vectorize(self,x_train,x_test,y_train,y_test, clean_data_frame):
		ngram_range = (1,2)
		min_df = 10
		max_df = 1.
		max_features = 7333

		tfidf = TfidfVectorizer(encoding='utf-8',
			                    ngram_range = ngram_range,
			                    stop_words = None,
			                    lowercase = False,
			                    max_df=max_df,
			                    min_df=min_df,
			                    max_features=max_features,
			                    norm='l2',
			                    sublinear_tf=True)
		features_train = tfidf.fit_transform(x_train).toarray()
		labels_train = y_train
		print(features_train.shape)

		with open('tfidf_transformer.pickle', 'wb') as output:
			pickle.dump(tfidf, output)

		with open('features_train.pickle', 'wb') as output:
			pickle.dump(features_train, output)

		with open('labels_train.pickle', 'wb') as output:
			pickle.dump(labels_train, output)

		with open('tfidf.pickle', 'wb') as output:
			pickle.dump(tfidf, output)

		features_test = tfidf.transform(x_test).toarray()
		labels_test = y_test
		print(features_test.shape)

		with open('features_test.pickle', 'wb') as output:
			pickle.dump(features_test, output)

		with open('labels_test.pickle', 'wb') as output:
			pickle.dump(labels_test, output)

		return features_train, labels_train,features_test, labels_test, clean_data_frame

		# we will use chi square to see unigrams and bigrams

		# for Product, category_id in sorted(clean_data_frame['category_code'].items()):
		# 	features_chi2 = chi2(features_train, labels_train == category_id)
		# 	indices = np.argsort(features_chi2[0])
		# 	feature_names = np.array(tfidf.get_feature_names())[indices]
		# 	unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
		# 	bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
		# 	print("# '{}' category:".format(Product))
		# 	print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
		# 	print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
		# 	print("")

		# # TF-IDF object
		# with open('tfidf.pickle', 'wb') as output:
		#     pickle.dump(tfidf, output)

	def main(self):
		preprocessing = PreProcessing()
		df = preprocessing.main()
		normalize = self.class_normalize(df)
		stop_words = self.stopwords_stemming(normalize)
		lemmatize = self.class_Lemma(stop_words)
		x_train,x_test,y_train,y_test, clean_data_frame = self.partition(lemmatize)
		features_train, labels_train,features_test, labels_test, clean_data_frame = self.vectorize(x_train,x_test,y_train,y_test,clean_data_frame)

		with open('clean_data_frame.pickle', 'wb') as output:
			pickle.dump(clean_data_frame, output)

		return features_train, labels_train,features_test, labels_test, clean_data_frame

if __name__ == "__main__":
	fe = FeatureEngineering()
	fe.main()
