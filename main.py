import csv_helper
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

#PART 1 Pre-Processing
#load the csv file
tweets = csv_helper.CSVHelper.load_csv("Dataset\Tweets_2016London.csv")
print(tweets)

#make tokenizer object
tknzr = TweetTokenizer()
#make list of tokens instead of list of tweets
tokenizedTweets = [tknzr.tokenize(i) for i in tweets]
print(tokenizedTweets)

#remove stop words (to get these stopwords open python console and run:
#import nltk
#nltk.download("stopwords")
stopwords = stopwords.words('english')
print(stopwords)
#remove every word from tokenized tweets which is in stopwords (keep the rest)


#works, but inefficient
filteredTweets = tokenizedTweets
for tweet in filteredTweets:
    for word in tweet:
        if word in stopwords:
            tweet.remove(word)
print(filteredTweets)