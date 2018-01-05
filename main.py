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

#DBScan v0.000001
X = tweetsTF

#minPts set to 10 (constant)
minPts = 10

#matrix to store results
dbScanResults = np.zeros((len(flatstemmedTweets), 35))

#iterate eps from 0.95 to 1.3 in steps of 0.1 (decided based on silhouette score and amount of clusters)
for i in range(95, 130, 1):
    eps = i/100
    core_points, cluster_labels, outliers = dbscan(X, eps, minPts)

    #TODO store results in dbSCANresult matrix, perform check afterwards
    print("EPS: ", eps)
    print('%d clusters found' % (len(core_points)))
    print('%d outlier points detected' % (len(outliers)))
    # Calculate the shlhouette score
    if len(core_points) > 1:
        print('Silhouette score: %f' % silhouette_score(X, cluster_labels))
    else:
        print('Cannot evaluate silhouetter score with only one cluster')


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

