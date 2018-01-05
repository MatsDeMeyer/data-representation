
import numpy as np

from sklearn.cluster import DBSCAN

def runDBSCAN(tfidf_matrix, min_samples, eps_min, eps_max, interval):
    print('############################################################################################')
    print('Running DBSCAN iteratively with a constan min samples of %d and an eps from %.2f to %.2f' % (min_samples, eps_min, eps_max))
    for eps in np.arange(eps_min, eps_max, interval):

        # DBSCAN algorithm from scikit learn
        print('DBSCAN with eps= %.2f' % eps)
        db = DBSCAN(eps, min_samples)
        tweet_db = db.fit_predict(tfidf_matrix)
        labels = db.labels_
        # Get every cluster
        clusters = set(labels)
        n_clusters = len(clusters)-1
        print('%d clusters found' % n_clusters)
        # If a cluster is not -1 (-1 = noise), convert to zero (border point)
        labels = np.where(labels != -1, 0, -1)

        # set core samples to 1
        core_samples_mask = np.zeros_like(db.labels_)
        core_samples_mask[db.core_sample_indices_] = 1
        labels[db.core_sample_indices_] = 1

        print('Sample of 40 tweets: -1=noise, 0=border point, 1=core point')
        print(labels[:40])

        noise = len(np.nonzero(labels==-1)[0])
        print('%d Noise points' % noise)
        noise_decision_labels = np.asmatrix(labels)
        noise_decision_labels = noise_decision_labels.reshape(-1, 1)

        # Completing the noise decision matrix
        if eps > eps_min:
            resultMatrix = np.hstack((resultMatrix, noise_decision_labels))
        else:
            resultMatrix = noise_decision_labels

    print('Sample of the results (10x10): -1=noise, 0=border point, 1=core point')
    print(resultMatrix[:10, :10])
    print('############################################################################################')
    return resultMatrix

#Get indices of noisy tweets
def getNoiseIndices(dbscanResults):
    #threshold at 50% (>50% border or noise point = noisy tweet)
    threshold = 0.5
    #amount of runs =
    runs = dbscanResults.shape[1]
    #get amount of border points
    borderTweets = (runs - np.count_nonzero(dbscanResults, axis=1))/runs
    #add 1 to every value, sets all noise tweets to 0
    dbscanResults = dbscanResults+1
    #get amount of noise points
    noiseTweets = (runs - np.count_nonzero(dbscanResults, axis=1))/runs
    noiseMask = np.hstack((noiseTweets, borderTweets))
    #get indices where amount of noise or border > threshold
    noiseIndices = np.nonzero(noiseMask > threshold)[0]
    #amount of noisy tweets
    n_noise = len(noiseIndices)
    print(n_noise, ' noise tweets detected with dbscan')
    return noiseIndices

#Remove noisy tweets from tweet set
def removeNoiseTweets(tweetsToClean, noiseIndices):
    cleanedTweets =np.delete(tweetsToClean, noiseIndices, 0)
    print('After removing the noisy tweets, %d tweets remain' % len(cleanedTweets))
    return cleanedTweets
