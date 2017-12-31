import csv_helper

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#PART 1: Pre-Processing
#load the csv file
tweets = csv_helper.CSVHelper.load_csv("Dataset\Tweets_2016London.csv")

#make tokenizer object
tknzr = TweetTokenizer()
#make list of tokens instead of list of tweets
tokenizedTweets = [tknzr.tokenize(i) for i in tweets]
print("Tokenized Tweets: ", tokenizedTweets)

#remove stop words (to get these stopwords uncomment the following lines (only has to be run once)
#import nltk
#nltk.download("stopwords")
stopwords = stopwords.words('english')
#remove every word from tokenized tweets which is in stopwords (keep the rest)
filteredTweets = [[word for word in tweet if word not in stopwords] for tweet in tokenizedTweets]
print("Filtered Tweets: ", filteredTweets)

#Stemming
st = SnowballStemmer("english")
stemmedTweets = [[st.stem(word) for word in tweet if word not in stopwords] for tweet in tokenizedTweets]
print("Stemmed Tweets: ", stemmedTweets)

#PART 2: Noise Removal
#TF-IDF (werkt nog niet)
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py

#vectorizer = TfidfVectorizer(max_df=0.99, max_features=10000, min_df=2, stop_words='english', use_idf=True)
#result = vectorizer.fit_transform(stemmedTweets)
#result = [vectorizer.fit_transform(tweet) for tweet in stemmedTweets]

flatstemmedTweets = [" ".join(tweets) for tweets in stemmedTweets]
print("Stemmed Tweets flattened: ", flatstemmedTweets)

vectorizer = TfidfVectorizer(min_df=1)
tweetsTF = vectorizer.fit_transform(flatstemmedTweets)



