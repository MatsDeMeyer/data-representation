import csv_helper
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#PART 1 Pre-Processing
#load the csv file
tweets = csv_helper.CSVHelper.load_csv("Dataset\Tweets_2016London.csv")

#make tokenizer object
tknzr = TweetTokenizer()
#make list of tokens instead of list of tweets
tokenizedTweets = [tknzr.tokenize(i) for i in tweets]
print("Tokenized Tweets: ", tokenizedTweets)

#remove stop words (to get these stopwords open python console and run:
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
