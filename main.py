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
#TF-IDF
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py

#vectorizer = TfidfVectorizer(max_df=0.99, max_features=10000, min_df=2, stop_words='english', use_idf=True)

flatstemmedTweets = [" ".join(tweets) for tweets in stemmedTweets]
print("Stemmed Tweets flattened: ", flatstemmedTweets)

vectorizer = TfidfVectorizer(min_df=1)
tweetsTF = vectorizer.fit_transform(flatstemmedTweets)

#k-means clustering
def euclidean_vectorized(A, B):
    n, d = A.shape
    m, d1 = B.shape
    A = A.toarray()
    B = B.toarray()

    assert d == d1, 'Incompatible shape'
    A_squared = np.sum(np.square(A), axis=1, keepdims=True)
    B_squared = np.sum(np.square(B), axis=1, keepdims=True)
    AB = np.matmul(A, B.T)
    distances = np.sqrt(A_squared - 2 * AB + B_squared.T)
    return distances


# X: data matrix of size (n_samples,n_features)
# n_clusters: number of clusters
# output 1: labels of X with size (n_samples,)
# output 2: centroids of clusters
def kmeans(X, n_clusters):
    # initialize labels and prev_labels. prev_labels will be compared with labels to check if the stopping condition
    # have been reached.
    prev_labels = np.zeros(X.shape[0])
    labels = np.zeros(X.shape[0])

    # init random indices
    # YOUR CODE GOES HERE
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)  # np.random.permutation(X.shape[0])[:n_clusters]

    # assign centroids using the indices
    # YOUR CODE GOES HERE
    centroids = X[indices]

    # the interative algorithm goes here
    while (True):
        # calculate the distances to the centroids
        # YOUR CODE GOES HERE
        distances = euclidean_vectorized(X, centroids)

        # assign labels
        # YOUR CODE GOES HERE
        labels = np.argmin(distances, axis=1)

        # stopping condition
        # YOUR CODE GOES HERE
        if np.array_equal(labels, prev_labels):
            # if np.sum(labels != prev_labels) == 0:
            break

        # calculate new centroids
        # YOUR CODE GOES HERE
        for cluster_indx in range(centroids.shape[0]):
            members = X[labels == cluster_indx]
            centroids[cluster_indx, :] = np.mean(members, axis=0)

        # keep the labels for next round's usage
        # YOUR CODE GOES HERE
        prev_labels = np.argmin(distances, axis=1)

    return labels, centroids

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
                if i != j:
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
#DBScan


#Part 3: Clustering
#first create new matrix of TF-IDF data without the tweets marked as noise
tweetMatrix = tweetsTF.toarray()
tweetsTF_K = tweetMatrix[kmeans_noise]

print(tweetsTF_K.shape)


