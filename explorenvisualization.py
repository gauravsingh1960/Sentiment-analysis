import pickle
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
from data_cleaning import PreProcessing

with open("features_train.pickle", 'rb') as output :
	features_train = pickle.load(output)

with open("labels_train.pickle", 'rb') as output :
	labels_train = pickle.load(output)

with open("features_test.pickle", 'rb') as output :
	features_test = pickle.load(output)

with open("labels_test.pickle", 'rb') as output :
	labels_test = pickle.load(output)


class Visualization:

	def Countplot(self, clean_data_frame):
		print(clean_data_frame["Sentiment Class"].value_counts())
		sns.countplot(x="Sentiment Class",data=clean_data_frame)
		plt.show()

	def Barplot(self,features_train, labels_train,features_test, labels_test,clean_data_frame):
		fig, ax = plt.subplots()
		data = clean_data_frame["Review_Rating"].value_counts()
		points = data.index
		frequency = data.values 

		ax.bar(points, frequency, color ='bgyr')
		ax.set_title('review ratings')
		ax.set_xlabel('points')
		ax.set_ylabel('frequency')
		plt.show()

	def Wordcloud(self,features_train, clean_data_frame):		
		#word cloud frequencies of words
		words = clean_data_frame[['Review_Text']].groupby(['Sentiment Class'] == 'Positive')
		wordcloud = WordCloud(max_font_size=40,max_words = 50,background_color = "white").fit_words(words)  #  fit_words() is used to plot wordcloud using dictionary.
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.show()

	def main(self):
		preprocessing = PreProcessing()
		clean_data_frame = preprocessing.main()
		Countplot  = self.Countplot(clean_data_frame)
		Barplot  = self.Barplot(features_train, labels_train,features_test, labels_test,clean_data_frame)
		# cloud = self.Wordcloud(features_train, clean_data_frame)

if __name__ == "__main__":
	visual = Visualization()
	visual.main()




