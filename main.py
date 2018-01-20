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
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from dbscan import *

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

flatstemmedTweets = [" ".join(tweets) for tweets in stemmedTweets]
print("Stemmed Tweets flattened: ", flatstemmedTweets)

#PART 2: Noise Removal

#TF-IDF representation
vectorizer = TfidfVectorizer(min_df=1)
tweetsTF = vectorizer.fit_transform(flatstemmedTweets)

#DBScan
#function using DBSCAN function from scikit-learn
#tfidf matrix = tweetsTF
#min_samples = 12
#eps min = 1.3
#eps_max = 1.4
#interval = 0.01 -> goes from 1.3 to 1.4 in steps of 0.01
dbscanResults = runDBSCAN(tweetsTF, 12, 1.3, 1.4, 0.01)

dbscanNoiseIndices = getNoiseIndices(dbscanResults)
print("DBSCAN Noiseindices: ", dbscanNoiseIndices)

dbscanCleanTweets = removeNoiseTweets(flatstemmedTweets, dbscanNoiseIndices)
print("First 20 Cleaned Tweets by DBSCAN: ", dbscanCleanTweets[0:20])

##K-MEANS
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

#Part 3: Clustering
#with k-means
#first create new matrix of TF-IDF data without the tweets marked as noise
print("***** KMEANS clustering *****")
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
#Get edges
#if a value in the consensusmatrix exceeds a certain threshold, make it an edges

edges = np.array([[0, 0]])
threshold = 12

print("check edges")
for i in range (0,consensusMatrix_K_filt.shape[0]):
    for j in range(0, np.math.ceil(consensusMatrix_K_filt.shape[0]/2)):
        if(consensusMatrix_K_filt[i][j])> threshold:
            edges = np.append(edges, [[i, j]], axis=0)

edges = np.delete(edges, (0), axis=0)



#print("edges:")
#print(edges)

#np.savetxt("edges.csv", edges, '%d', delimiter=",")


#finally run kmeans one more time with the consensusmatrix as input

cluster_labels_kmeans, centroids_kmeans = kmeans(consensusMatrix_K_filt, 9)


#identify most common words from kmeans

#9 indexes of most common words will be stored in this list
index_word_kmeans = list()

for i in range (0,9):
    #indexes of the tweets with a certain number
    index_tweets = np.nonzero(cluster_labels_kmeans == i)[0]
    #get all those tweets
    tweets_singleCluster = tweetsTF_K[index_tweets]
    #check collumn for most important word
    mostCommon = np.sum(tweets_singleCluster, axis=0).argmax()
    index_word_kmeans.append(mostCommon)


print("indices words kmeans", index_word_kmeans)

words = vectorizer.get_feature_names()
for i in range (0, 9):
    print(words[index_word_kmeans[i]])


#with dbscan
print("***** DBSCAN clustering *****")
#tfidf representation so we can run k-means
tweetsTF_dbscan = vectorizer.fit_transform(dbscanCleanTweets)

#Create consensusmatrix for filtered dbscan data

consensusMatrix_K_filt_dbscan = np.zeros((tweetsTF_dbscan.shape[0],tweetsTF_dbscan.shape[0]))

for i in range (2,12):
    num_clusters = i
    cluster_labels, centroids = kmeans(tweetsTF_dbscan, num_clusters)

    for j in range (1, tweetsTF_dbscan.shape[0]):
        for k in range (1, tweetsTF_dbscan.shape[0]):
            if (cluster_labels[j] == cluster_labels[k]):
                if j != k:
                    consensusMatrix_K_filt_dbscan[j][k] += 1
                    consensusMatrix_K_filt_dbscan[k][j] += 1
                else:
                    consensusMatrix_K_filt_dbscan[j][k] += 1

#finally run kmeans one more time with the consensusmatrix as input

cluster_labels_dbscan, centroids_dbscan = kmeans(consensusMatrix_K_filt_dbscan, 9)

#identify most common words from dbscan

#9 indexes of most common words will be stored in this list
index_word_dbscan = list()

for i in range (0,9):
    #indexes of the tweets with a certain number
    index_tweets = np.nonzero(cluster_labels_dbscan == i)[0]
    #get all those tweets
    tweets_singleCluster = tweetsTF_dbscan[index_tweets]
    #check collumn for most important word
    mostCommon = np.sum(tweets_singleCluster, axis=0).argmax()
    index_word_dbscan.append(mostCommon)


print("indices words dbscan", index_word_dbscan)

for i in range (0, 9):
    print(words[index_word_dbscan[i]])

