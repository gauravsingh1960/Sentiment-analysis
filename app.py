from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
model = pickle.load(open('df_models_mnbc.pickle','rb'))

with open("clean_data_frame.pickle", 'rb') as output :
	clean_data_frame = pickle.load(output)

@app.route('/')
def home():
	return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

	# df= pd.read_csv("train_data_20000.csv", encoding="latin-1")
	# df.drop(['Review_Rating'], axis=1, inplace=True)
	# Features and Labels
	clean_data_frame['label'] = clean_data_frame['Sentiment Class'].map({'Positive':0, 'Negative':1,'Neutral':2})
	X = clean_data_frame['Review_Text']
	y = clean_data_frame['label']
	
	# # Extract Feature With CountVectorizer
	ngram_range = (1,2)
	min_df = 10
	max_df = 1.
	max_features = 300

	tfidf = TfidfVectorizer(encoding='utf-8',
			                ngram_range = ngram_range,
			                stop_words = None,
			                lowercase = False,
			                max_df=max_df,
			                min_df=min_df,
			                max_features=max_features,
			                norm='l2',
			                sublinear_tf=True)

	cv = TfidfVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	# Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['Review_Text']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)

	if my_prediction == 0:
		return render_template("index.html",predicted="Positive")
	
	elif my_prediction == 1:
		return render_template("index.html",predicted="Negative")

	else:
		return render_template("index.html",predicted="Neutral")

if __name__ == '__main__':
	app.run(debug=True)
