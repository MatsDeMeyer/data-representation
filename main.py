import csv_helper
from nltk.tokenize import TweetTokenizer

#load the csv file
tweets = csv_helper.CSVHelper.load_csv("Dataset\Tweets_2016London.csv")
print(tweets)

#make tokenizer object
tknzr = TweetTokenizer()
#make list of tokens instead of list of tweets
tokenizedTweets = [tknzr.tokenize(i) for i in tweets]
print(tokenizedTweets)