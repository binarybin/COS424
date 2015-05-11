###############################################################################
# graph.py
# Authors: Adam Fisch, Bin Xu, and Ian Leng
# COS 424 Final Project
#
# Description: Construction of the bidder-auction graph for data analysis of
#              the latent structure. Sets up foundations for feeding to
#              further algorithms.
###############################################################################
import numpy as np
import scipy.sparse as sp
from pandas import DataFrame as df
from collections import Counter
from optparse import OptionParser
from os.path import join
import time, sys
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

BIDS_FILE = 'bids.csv'
BIDDER_FILE = 'train.csv'
TEST_FILE = 'test.csv'

###############################################################################
# Main. Read in data and run analysis.
###############################################################################
def main(argv):
    # Get the path and read files.
    parser = OptionParser()
    parser.add_option("-p", "--path", dest = "path",
                      help = 'read data from PATH', metavar = 'PATH')
    (options, _args) = parser.parse_args()
    path = ''
    if options.path:
        path = options.path
    print "PATH = " + path
    start_time = time.time()
    # Read in auction file.
    print("Reading in the auction file...")
    (bidders, auctions, bids) = read_auction_data(path)

    # Create sparse matrix from auction data
    print("Creating sparse matrix...")
    (Xsparse, bidder2id, id2bidder, auction2id, id2auction) = \
        create_sparse_matrix(bidders, auctions, bids)
    print("Number of entries = %s" % len(Xsparse.data))

    # Read in the bidder data and labels
    print("Reading in the training file...")
    (bidder_list, labels) = read_bidder_labels(path)

    # Filter non-interacting bidders
    (bidder_list, labels) = filter_non_interacting(bidder_list,
        bidder2id, labels)

    # Separate the robot and human data
    print("Separating humans and robots...")
    print("Calculating robots...")
    robots_labels = np.where(labels > 0)[0]
    (robots, r_labels) = filter_non_interacting(bidder_list[robots_labels],
        bidder2id, labels[robots_labels])
    robot_indices = [bidder2id.get(item, item) for item in robots]
    print("Number of known robots: %s" % len(robot_indices))

    print("Calculating humans...")
    humans_labels = np.where(labels == 0)[0]
    (humans, h_labels) = filter_non_interacting(bidder_list[humans_labels],
        bidder2id, labels[humans_labels])
    human_indices = [bidder2id.get(item, item) for item in humans]
    print("Number of known humans: %s" % len(human_indices))

    robot_sparse = Xsparse.tocsr()[robot_indices, :]
    human_sparse = Xsparse.tocsr()[human_indices, :]

    # Visualize the distribution of the number of bids placed per auction,
    # separated by robots vs humans.
    plt.figure()
    (n_r, bins_r, patches_r) = plt.hist(robot_sparse.data, 50, alpha = 0.5,
                                        normed = 1, label = 'robot')
    plt.hist(human_sparse.data, bins = bins_r, alpha = 0.5,
             normed = 1, label = 'human')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.title('Distribution of Robot vs. Human Bids Placed Per Auction')
    plt.xlabel('Number of Bids in One Auction')
    plt.ylabel('Density')

    # Visualize the distribution of the total number of bids placed by bidder,
    # separated by robots vs humans.
    plt.figure()
    (n_r, bins_r, patches_r) = plt.hist(robot_sparse.sum(1), 50, alpha = 0.5,
                                        normed = 1, label = 'robot')
    plt.hist(human_sparse.sum(1), bins = bins_r, alpha = 0.5,
             normed = 1, label = 'human')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.title('Distribution of Robot vs. Human Total Bids')
    plt.xlabel('Number of Total Bids Placed')
    plt.ylabel('Density')

    # Compute the details on a SVD decomposition - the explained variance
    # ratios, the singular values themselves, and a 2D projection for plotting.
    (variance_ratios, X3D) = svdData(Xsparse)
    plt.figure()
    plt.plot(variance_ratios, 'bo-')
    plt.title('Singular Values: Decreasing Explained Variance')
    plt.xlabel('Dimension Number')
    plt.ylabel('Explained Variance')

    fig = plt.figure()
    robots3d = X3D[robot_indices, :]
    humans3d = X3D[human_indices, :]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(robots3d[:,0],robots3d[:,1],robots3d[:,2],
        color = 'b', label = 'robot')
    ax.scatter(humans3d[:,0],humans3d[:,1],humans3d[:,2],
        color = 'g', label = 'human')
    ax.set_title('3D Projection of Bid Data')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    # Binarize matrix and now only consider participation in auction
    # Visualize number of auctions robots participate in vs humans.
    robot_sparse_binary = robot_sparse.copy()
    human_sparse_binary = human_sparse.copy()
    robot_sparse_binary.data[:] = 1
    human_sparse_binary.data[:] = 1
    plt.figure()
    (n_r, bins_r, patches_r) = plt.hist(robot_sparse_binary.sum(1), 50,
        alpha = 0.5, normed = 1, label = 'robot')
    plt.hist(human_sparse_binary.sum(1), bins = bins_r, alpha = 0.5,
        normed = 1, label = 'human')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.title('Distribution of Robot vs. Human Auction Participation')
    plt.xlabel('Number of Unique Auctions')
    plt.ylabel('Density')

    # Create training and testing matrix
    bidder_indices = [bidder2id.get(item, item) for item in bidder_list]
    Xsparse_reduced = doSVD(Xsparse, 15)
    Xtrain_reduced = Xsparse_reduced[bidder_indices, :]
    le = LabelEncoder()
    le.fit(labels)
    ylabel = le.transform(labels)

    adaBoost = AdaBoostClassifier(n_estimators = 500, learning_rate = 1.95)
    (mean_fpr_ab, mean_tpr_ab) = kFoldROC('AdaBoost', adaBoost,
                                           Xtrain_reduced, ylabel, 5)

    knn = KNeighborsClassifier(n_neighbors=50)
    (mean_fpr_knn, mean_tpr_knn) = kFoldROC('KNN', knn,
                                             Xtrain_reduced, ylabel, 5)

    rf = RandomForestClassifier(n_estimators = 500)
    (mean_fpr_rf, mean_tpr_rf) = kFoldROC('Random Forrest', rf,
                                           Xtrain_reduced, ylabel, 5)

    # Calculates test labels for submission
    print("Running on test data...")
    test_bidder_list = read_test_bidder(path)
    filt_test_list = filter_replace_test(test_bidder_list, bidder2id, id2bidder)
    test_indices = [bidder2id.get(item, item) for item in filt_test_list]
    Xtest = Xsparse_reduced[test_indices, :]
    rf = RandomForestClassifier(n_estimators = 500)
    probs = rf.fit(Xtrain_reduced, ylabel).predict_proba(Xtest)

    # Show everything
    plt.show()
    print time.time() - start_time

###############################################################################
# A helper method to evaluate kFold cross-validation
###############################################################################
def kFoldROC(name, classifier, Xtrain, ytrain, folds):
    cv = StratifiedKFold(ytrain, n_folds = folds)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    plt.figure()
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(Xtrain[train],
                     ytrain[train]).predict_proba(Xtrain[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(ytrain[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1,
            label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Charateristic for %s' % name)
    plt.legend(loc="lower right")
    return (mean_fpr, mean_tpr)

def outputPredictions(classifier, Xtest, Xtrain, ytrain):
    probas_ = classifer.fit(Xtrain, ytrain).predict_proba(Xtest)

###############################################################################
# Helper method to run a single SVD decomposition, selecting only n
# eigenvectors. Returns the reconstructed, test and train "vectors."
###############################################################################
def doSVD(Xtrain, n):
    svd = TruncatedSVD(n_components = n)
    svd.fit(Xtrain)
    Xtrain_reduced = svd.fit_transform(Xtrain)
    return (Xtrain_reduced)

###############################################################################
# Helper method to run a SVM classification.
###############################################################################
def support_vector(XTrain, yTrain, XTest):
    svm = SVC(kernel='linear',probability = True)
    svm.fit(XTrain, yTrain)
    scores = svm.predict_proba(XTest)
    labels = svm.predict(XTest)

    return (labels, scores)

###############################################################################
# A helper method to extract relevant data from the singular value
# decomposition approach. Returns the 30 largest singular values, their
# explained variances, and a 2 dimensional projection of the data
# (for easy investigative plotting).
###############################################################################
def svdData(Xtrain):
    svd_variance = TruncatedSVD(n_components = 30)
    svd_variance.fit(Xtrain)
    variance_ratios = svd_variance.explained_variance_ratio_
    svd = TruncatedSVD(n_components = 3)
    svd.fit(Xtrain)
    X3D = svd.fit_transform(Xtrain)
    return (variance_ratios, X3D)

###############################################################################
# A helper method to read in the original csv auction data file.
# Returns three ordered lists:
#   - list of bidders (with duplicates)
#   - list of auctions (with duplicates)
#   - number of bids bidder i placed in auction i
#
# In sparse matrix notation this is (bidder, auction) = bids.
###############################################################################
def read_auction_data(path):
    csv = df.from_csv(join(path, BIDS_FILE))
    bids_mat = csv.as_matrix()
    auctions_pairs = zip(bids_mat[:,0], bids_mat[:,1])
    counts = Counter(auctions_pairs)
    data = np.array(counts.items())
    bidders, auctions = np.array(zip(*data[:,0]))
    bids = np.array(data[:,1], dtype = float)

    print ("Bidders shape: %s" % bidders.shape)
    print ("Auctions shape: %s" % auctions.shape)
    print ("Bids shape: %s" % bids.shape)
    return (bidders, auctions, bids)

###############################################################################
# A helper method to read in the training csv data file.
# Returns two ordered lists:
#   - list of bidders (unique ids, no duplicate)
#   - list of labels (0 or 1 for not robot / is robot)
###############################################################################
def read_bidder_labels(path):
    csv = df.from_csv(join(path, BIDDER_FILE), index_col = False)
    bidder_mat = csv.as_matrix()
    bidder_list = bidder_mat[:, 0]
    labels = np.array(bidder_mat[:,3], dtype = float)

    return (bidder_list, labels)

###############################################################################
# A helper method to read in the test csv data file for Kaggle submission.
# Returns one list:
#   - list of bidders (unique ids, no duplicate)
###############################################################################
def read_test_bidders(path):
    csv = df.from_csv(join(path, TEST_FILE), index_col = False)
    bidder_mat = csv.as_matrix()
    test_list = bidder_mat[:, 0]

    return (test_list)

###############################################################################
# A helper method to create a numerical sparse COO format matrix from the
# bid data. Returns:
#   - a numerical sparse matrix of (bidder, auction) = bids
#   - a mapping from a numerical id to corresponding string bidder id
#   - a mapping from a numerical id to corresponding string auction id
###############################################################################
def create_sparse_matrix(bidders, auctions, bids):
    # Mapping dictionaries
    bidder2id = {}
    id2bidder = {}
    auction2id = {}
    id2auction = {}

    # Map bidder strings to numbers from 0 to N, where N is number of bidders
    count = 0
    for i in bidders:
        if (i not in bidder2id):
            bidder2id[i] = count
            id2bidder[count] = i
            count = count + 1
    print('Number of bidder ids: %s' % count)
    dim_bidders = count
    bidders_ids = [bidder2id.get(item, item) for item in bidders]

    # Map auction strings to numbers from 0 to M, where M is number of auctions
    count = 0
    for i in auctions:
        if (i not in auction2id):
            auction2id[i] = count
            id2auction[i] = i
            count = count + 1
    print('Number of auction ids: %s' % count)
    dim_auctions = count
    auctions_ids = [auction2id.get(item, item) for item in auctions]

    # Create the sparse matrix
    matrix = sp.coo_matrix((bids, (bidders_ids, auctions_ids)),
                            shape=(dim_bidders,dim_auctions))

    return (matrix, bidder2id, id2bidder, auction2id, id2auction)

###############################################################################
# A helper method to filter out bidders that do not appear in the auction
# data.
###############################################################################
def filter_non_interacting(bidders, bidder2id, labels):
    filtered_bidders = []
    filtered_labels = []
    count_filtered = 0
    for i, j in enumerate(bidders):
        if (j in bidder2id):
            filtered_bidders.append(j)
            filtered_labels.append(labels[i])
        else:
            count_filtered = count_filtered + 1
    print("Filtered %s bidders." % count_filtered)

    return (np.array(filtered_bidders), np.array(filtered_labels))

###############################################################################
# A helper method to filter out test bidders that do not appear in the auction
# data. Replaces those bidders with a dummy duplicate bidder (for submission
# compatibility).
###############################################################################
def filter_replace_test(bidders, bidder2id, id2bidder):
    filtered_test_bidders = []
    count_filtered = 0
    for i in bidders:
        if (i in bidder2id):
            filtered_test_bidders.append(i)
        else:
            filtered_test_bidders.append(id2bidder[0])
            count_filtered = count_filtered + 1
    print("Filtered %s bidders." % count_filtered)

    return (np.array(filtered_bidders))

###############################################################################
# Calls main when class is directly invoked.
###############################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
