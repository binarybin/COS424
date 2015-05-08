###############################################################################
# graph.py
# Authors: Adam Fisch, Bin Xu, and Ian Leng
# COS 424 Final Project
#
# Description: Graph approach to analyzing the robot-human auction system.
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

BIDS_FILE = 'bids.csv'
BIDDER_FILE = 'train.csv'

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
    svd = TruncatedSVD(n_components = 2)
    svd.fit(Xtrain)
    X2D = svd.fit_transform(Xtrain)
    return (variance_ratios, X2D)

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
def filter_non_interacting(bidders, bidder2id):
    filtered_bidders = []
    count_filtered = 0
    for i in bidders:
        if (i in bidder2id):
            filtered_bidders.append(i)
        else:
            count_filtered = count_filtered + 1
    print("Filtered %s bidders." % count_filtered)

    return (filtered_bidders)

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

    # Separate the robot and human data
    print("Separating humans and robots...")
    print("Calculating robots...")
    robots_labels = np.where(labels > 0)[0]
    robots = filter_non_interacting(bidder_list[robots_labels], bidder2id)
    robot_indices = [bidder2id.get(item, item) for item in robots]
    print("Number of known robots: %s" % len(robot_indices))

    print("Calculating humans...")
    humans_labels = np.where(labels == 0)[0]
    humans = filter_non_interacting(bidder_list[humans_labels], bidder2id)
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
    plt.xlabel('Number of Total Bids Places')
    plt.ylabel('Density')

    # Compute the details on a SVD decomposition - the explained variance
    # ratios, the singular values themselves, and a 2D projection for plotting.
    (variance_ratios, X2d) = svdData(Xsparse)
    plt.figure()
    plt.plot(variance_ratios, 'bo-')
    plt.title('Singular Values: Decreasing Explained Variance')
    plt.figure()
    plt.xlabel('Dimension Number')
    plt.ylabel('Explained Variance (%)')
    plt.scatter(X2d[:,0], X2d[:,1])
    plt.title('2D Projection of Transaction Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    print time.time() - start_time
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
