import csv_helper

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from kmeans import *
from sklearn.cluster import DBSCAN

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
#TF-IDF
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py

#vectorizer = TfidfVectorizer(max_df=0.99, max_features=10000, min_df=2, stop_words='english', use_idf=True)

flatstemmedTweets = [" ".join(tweets) for tweets in stemmedTweets]
print("Stemmed Tweets flattened: ", flatstemmedTweets)

vectorizer = TfidfVectorizer(min_df=1)
tweetsTF = vectorizer.fit_transform(flatstemmedTweets)

#Creating consensus matrix:
#process the tweets with several values for k with the k-means algorithm
#increment the value of the coördinates for each tweet that got clustered together
consensusMatrix_K = np.zeros((2001,2001))

for i in range (2,12):
    num_clusters = i
    cluster_labels, centroids = kmeans(tweetsTF, num_clusters)

    for j in range (1, 2001):
        for k in range (1, 2001):
            if (cluster_labels[j] == cluster_labels[k]):
                if j != k:
                    consensusMatrix_K[j][k] += 1
                    consensusMatrix_K[k][j] += 1
                else:
                    consensusMatrix_K[j][k] += 1

#if tweets didn't cluster together 10% of the time, set position to 0
consensusMatrix_K[consensusMatrix_K < 1] = 0

#mark tweets with cluster rate lower then threshold as noise
#determine threshold:

matrix_mean = np.sum(consensusMatrix_K, 1).mean()

kmeans_noise = list()

for i in range (0, consensusMatrix_K.shape[0]):
    if consensusMatrix_K[i, :].sum() > matrix_mean:
        kmeans_noise.append(i)

kmeans_noise = np.asarray(kmeans_noise)
print(kmeans_noise)
print(kmeans_noise.shape)

#DBScan v0.000001
# #############################################################################
# Compute DBSCAN

db = DBSCAN(eps=0.7, min_samples=3).fit(tweetsTF)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)




#Part 3: Clustering
#first create new matrix of TF-IDF data without the tweets marked as noise
tweetMatrix = tweetsTF.toarray()
tweetsTF_K = tweetMatrix[kmeans_noise]

#Create consensusmatrix for filtered kmeans data

consensusMatrix_K_filt = np.zeros((tweetsTF_K.shape[0],tweetsTF_K.shape[0]))

for i in range (2,12):
    num_clusters = i
    cluster_labels, centroids = kmeans(tweetsTF_K, num_clusters)

    for j in range (1, tweetsTF_K.shape[0]):
        for k in range (1, tweetsTF_K.shape[0]):
            if (cluster_labels[j] == cluster_labels[k]):
                if j != k:
                    consensusMatrix_K_filt[j][k] += 1
                    consensusMatrix_K_filt[k][j] += 1
                else:
                    consensusMatrix_K_filt[j][k] += 1

#finally run kmeans one more time with the consensusmatrix as input

cluster_labels, centroids = kmeans(consensusMatrix_K_filt, 9)

#identify most common words from kmeans

#9 indexes of most common words will be stored in this list
index_word = list()

for i in range (0,9):
    #indexes of the tweets with a certain number
    index_tweets = np.nonzero(cluster_labels == i)[0]
    #get all those tweets
    tweets_singleCluster = tweetsTF_K[index_tweets]
    #check collumn for most important word
    mostCommon = np.sum(tweets_singleCluster, axis=0).argmax()
    index_word.append(mostCommon)


print(index_word)

words = vectorizer.get_feature_names()
for i in range (0, 9):
    print(words[index_word[i]])

