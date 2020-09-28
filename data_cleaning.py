import pandas as pd
from nltk.tokenize import word_tokenize
import pickle

class PreProcessing:

	def read_data(self,raw_data_frame):
		df = pd.read_csv(raw_data_frame, delimiter= "," ,header=0, encoding= "utf")

		return df

	def lower_case(self,df):

		# shoes_reviews_df = self.read_data(raw_data_file_name)
		df["Review_Text"] = df["Review_Text"].str.lower()
		print(df["Review_Text"])

		return df

	def missing_value(self,df):

		print(df["Review_Text"].isnull().sum())
		df["Review_Text"] = df["Review_Text"].fillna("")

		df["Review_Rating"] = df["Review_Rating"].dropna()
		return df

	def remove_numeric_values(self,df):
		
		df["Review_Text"] = df["Review_Text"].str.replace('\\d+','')
		df["Review_Text"] = df["Review_Text"].str.replace('\n','')
		# print(df.head())			
		return df

	def remove_special_char(self,df):
		"""To remove special characters"""
		spec_chars = ["'s","!",'"',"#","%","&","'", "*","+",",","-",".","/",":",";","<","$","=",">","?","@","^","_","`","~","–"]
		for char in spec_chars:
			df['Review_Text'] = df['Review_Text'].str.replace(char,'')
		print(df['Review_Text'])

		with open('clean_data_frame.pickle', 'wb') as output:
			pickle.dump(df, output)

		return df

	def main(self):
		df = self.read_data("train_data_20000.csv")
		df = self.lower_case(df)
		df = self.missing_value(df)
		df = self.remove_numeric_values(df)
		df = self.remove_special_char(df)

		return df

if __name__ == "__main__":
	preprocessing = PreProcessing()
	preprocessing.main()










