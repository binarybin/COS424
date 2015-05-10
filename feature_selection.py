"""
Usage: python feature_selection.py auto <C> <kernel=linear, rbf, poly, sigmoid> <gamma>
"""


import pandas as pd
import numpy as np
from random import sample
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
import cPickle as pickle
from sys import argv

C = int(argv[2])
kernel = argv[3]
gamma = float(argv[4])


bid=pd.read_csv('bids.csv')
train=pd.read_csv('train.csv')
print "Data loaded"


def extract_feature(df):
    gb=df.groupby('bidder_id')
    #number of countries
    country_count=gb['country'].apply(pd.value_counts).count(level=0)
    #of unique device
    device_count=gb['device'].apply(pd.value_counts).count(level=0)
    # of unique ip
    ip_count=gb['ip'].apply(pd.value_counts).count(level=0)
    # of unique auction
    auction_count=gb['auction'].apply(pd.value_counts).count(level=0)
    # of unique url
    url_count=gb['url'].apply(pd.value_counts).count(level=0)
    # of transitions
    time_count=gb['time'].apply(pd.value_counts).count(level=0)
    #time interval **** zero value represents that only one transition has happened ****
    grouped=[gb.get_group(x) for x in gb.groups]
    rows_list=[]
    for i in range(0,len(grouped)):
        dict1={}
        dict1.update({'bidder_id':grouped[i]['bidder_id'].iloc[0],'bidderID':grouped[i]['bidder_id'].iloc[0],'time interval':grouped[i]['time'].iloc[len(grouped[i].index)-1]-grouped[i]['time'].iloc[0]})      
        rows_list.append(dict1)
    time = pd.DataFrame(rows_list)
    time_int=time.set_index('bidder_id')
    #average transition time: time interval/transitions
    pieces = [country_count,device_count,ip_count,auction_count,url_count,time_count]
    concatenated = pd.concat(pieces,axis=1,keys=['country', 'device','ip','auction','url','transitions'])
    concatenated = pd.concat([concatenated,time_int],axis=1)
    return concatenated

try:
    with open("features", 'rb') as fp:
        a = pickle.load(fp)
    print "Feature loaded!"

except:
    print "Feature loading failed, re-extracting!"
    a=extract_feature(bid)
    with open("features", 'wb') as fp:
        pickle.dump(a, fp)
    print "Feature extracted!"



train_size = 1200
rindex =  np.array(sample(xrange(len(train)), train_size))
real_train = train.ix[rindex]
real_test = train[~train.isin(real_train).all(1)]
trainable = a[a.bidderID.isin(real_train.bidder_id)]
testable = a[a.bidderID.isin(real_test.bidder_id)]

train_df = pd.concat([real_train.set_index('bidder_id'), trainable], axis=1)
train_df = train_df[np.isfinite(train_df['ip'])]
train_df = train_df.reset_index().drop("index", 1).drop("payment_account", 1).drop("address", 1).drop("bidderID",1)

test_df = pd.concat([real_test.set_index('bidder_id'), testable], axis=1)
test_df = test_df[np.isfinite(test_df['ip'])]
test_df = test_df.reset_index().drop("index", 1).drop("payment_account", 1).drop("address", 1).drop("bidderID",1)


train_x, train_y = train_df.values[:,1:], train_df.values[:,0].astype(int)
test_x, test_y = test_df.values[:,1:], test_df.values[:,0].astype(int)

print "Train data selected"

def Predict(model, train_x, train_y, test_x, test_y):
    detector = model.fit(train_x, train_y)
    predictions = detector.predict(test_x)
    print 'accuracy', accuracy_score(predictions, test_y)
    print 'confusion matrix\n', confusion_matrix(test_y, predictions)
    print '(row=expected, col=predicted)'
    return accuracy_score, confusion_matrix

def ROC(model, train_x, train_y, test_x, test_y):
    detector = model.fit(train_x, train_y)
    y_score = detector.predict_proba(test_x)
    fpr, tpr, _ = roc_curve(test_y, y_score[:,1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

if kernel == "linear":
    model = SVC(C = C,  kernel = kernel, probability = True)
else:
    model = SVC(C = C,  kernel = kernel, gamma = gamma, probability = True)

if argv[1] == "auto":
    filename = "SVM_"+kernel+"_C_"+str(C)+"_gamma_"+str(gamma)
else:
    filename = argv[1]

Prediction_result = Predict(model, train_x, train_y, test_x, test_y)

with open(filename+"PRED", 'wb') as fp:
    pickle.dump(Prediction_result, fp)
print "prediction finished"



ROC_result = ROC(model, train_x, train_y, test_x, test_y)

with open(filename+"ROC", 'wb') as fp:
    pickle.dump(ROC_result, fp)
print "ROC finished"

"""
# In[ ]:

ROC(SVC(C = 1,  kernel = "linear", probability = True), train_x, train_y, test_x, test_y)


# In[ ]:

ROC(SVC(C = 1000,  kernel = "rbf", gamma = 0.001, probability = True), train_x, train_y, test_x, test_y)


# In[ ]:

"""

