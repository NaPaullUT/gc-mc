""" Experiment runner for the model with knowledge graph attached to interaction data """


from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features, load_official_trainvaltest_split_cold_start
from model import RecommenderGAE, RecommenderSideInfoGAE
from utils import construct_feed_dict

# Set random seed
# seed = 123 # use only for unit testing


# Define parameters
DATASET = 'ml_100k'
DATASEED = 1234
NB_EPOCH = 1000
num_exp = 10
DO = 0.7
HIDDEN = [500, 75]
FEATHIDDEN = 10
BASES = 2
LR = 0.01
WRITESUMMARY = 0
SUMMARIESDIR = 'PATH'
FEATURES = 1
SYM = 0
TESTING = 0
ACCUM = 'stack'

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = False

C_START = True

N_R = 5
N_C = 5


NUMCLASSES = 5

# Splitting dataset in training, validation and test set
if FEATURES:
    datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
else:
    datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'


print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
if not C_START or N_C==0 or N_R==0:
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(DATASET, TESTING)
else: 
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split_cold_start(DATASET, TESTING,n_c=N_C,n_r=N_R)
    
print(train_labels)

num_users, num_items = adj_train.shape

num_side_features = 0

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features

    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)

    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]

    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

else:
    raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)


# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)

    if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
        sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)


support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

if SELFCONNECTIONS:
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))

num_support = len(support)
support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div

# Collect all user and item nodes for test set
test_u = list(set(test_u_indices))
test_v = list(set(test_v_indices))
test_u_dict = {n: i for i, n in enumerate(test_u)}
test_v_dict = {n: i for i, n in enumerate(test_v)}

test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

test_support = support[np.array(test_u)]
test_support_t = support_t[np.array(test_v)]

# Collect all user and item nodes for validation set
val_u = list(set(val_u_indices))
val_v = list(set(val_v_indices))
val_u_dict = {n: i for i, n in enumerate(val_u)}
val_v_dict = {n: i for i, n in enumerate(val_v)}

val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

val_support = support[np.array(val_u)]
val_support_t = support_t[np.array(val_v)]

# Collect all user and item nodes for train set
train_u = list(set(train_u_indices))
train_v = list(set(train_v_indices))
train_u_dict = {n: i for i, n in enumerate(train_u)}
train_v_dict = {n: i for i, n in enumerate(train_v)}

train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

train_support = support[np.array(train_u)]
train_support_t = support_t[np.array(train_v)]

# features as side info
if FEATURES:
    test_u_features_side = u_features_side[np.array(test_u)]
    test_v_features_side = v_features_side[np.array(test_v)]

    val_u_features_side = u_features_side[np.array(val_u)]
    val_v_features_side = v_features_side[np.array(val_v)]

    train_u_features_side = u_features_side[np.array(train_u)]
    train_v_features_side = v_features_side[np.array(train_v)]

else:
    test_u_features_side = None
    test_v_features_side = None

    val_u_features_side = None
    val_v_features_side = None

    train_u_features_side = None
    train_v_features_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

# create model
if FEATURES:
    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   logging=True)
else:
    model = RecommenderGAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_classes=NUMCLASSES,
                           num_support=num_support,
                           self_connections=SELFCONNECTIONS,
                           num_basis_functions=BASES,
                           hidden=HIDDEN,
                           num_users=num_users,
                           num_items=num_items,
                           accum=ACCUM,
                           learning_rate=LR,
                           logging=True)

# Convert sparse placeholders to tuples to construct feed_dict
test_support = sparse_to_tuple(test_support)
test_support_t = sparse_to_tuple(test_support_t)

val_support = sparse_to_tuple(val_support)
val_support_t = sparse_to_tuple(val_support_t)

train_support = sparse_to_tuple(train_support)
train_support_t = sparse_to_tuple(train_support_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, DO,
                                      train_u_features_side, train_v_features_side)
# No dropout for validation and test runs
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side)


# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()



if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

exp_dict = {}
for _ in range(num_exp):

    seed = int(time.time())
    np.random.seed(seed)
    tf.set_random_seed(seed)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    best_val_score = np.inf
    best_val_loss = np.inf
    best_epoch = 0
    wait = 0

    print('\nTraining...')
    epoch_dict = {
        't_er':[],
        'v_er':[]
    }
    for epoch in tqdm(range(NB_EPOCH)):

        t = time.time()

        # Run single weight update
        # outs = sess.run([model.opt_op, model.loss, model.rmse], feed_dict=train_feed_dict)
        # with exponential moving averages
        outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict)

        train_avg_loss = outs[1]
        train_rmse = outs[2]

        val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

        epoch_dict['t_er'].append(train_rmse)
        epoch_dict['v_er'].append(val_rmse)

        #store train_rmse and val_rmse here
        
        if VERBOSE:
            print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
                "train_rmse=", "{:.5f}".format(train_rmse),
                "val_loss=", "{:.5f}".format(val_avg_loss),
                "val_rmse=", "{:.5f}".format(val_rmse),
                "\t\ttime=", "{:.5f}".format(time.time() - t))

        if val_rmse < best_val_score:
            best_val_score = val_rmse
            best_epoch = epoch

        if epoch % 20 == 0 and WRITESUMMARY:
            # Train set summary
            summary = sess.run(merged_summary, feed_dict=train_feed_dict)
            train_summary_writer.add_summary(summary, epoch)
            train_summary_writer.flush()

            # Validation set summary
            summary = sess.run(merged_summary, feed_dict=val_feed_dict)
            val_summary_writer.add_summary(summary, epoch)
            val_summary_writer.flush()

        if epoch % 100 == 0 and epoch > 1000 and not TESTING and False:
            saver = tf.train.Saver()
            save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

            # load polyak averages
            variables_to_restore = model.variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, save_path)

            val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

            print('polyak val loss = ', val_avg_loss)
            print('polyak val rmse = ', val_rmse)

            # Load back normal variables
            saver = tf.train.Saver()
            saver.restore(sess, save_path)


    # store model including exponential moving averages
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "tmp/%s.ckpt" % model.name, global_step=model.global_step)


    if VERBOSE:
        print("\nOptimization Finished!")
        print('best validation score =', best_val_score, 'at iteration', best_epoch)


    if TESTING:
        test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
        print('test loss = ', test_avg_loss)
        print('test rmse = ', test_rmse)

        # restore with polyak averages of parameters
        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        #saver.restore(sess, save_path)

        test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
        print('polyak test loss = ', test_avg_loss)
        print('polyak test rmse = ', test_rmse)

    else:
        # restore with polyak averages of parameters
        #variables_to_restore = model.variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        #saver.restore(sess, save_path)

        #val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)
        #print('polyak val loss = ', val_avg_loss)
        #print('polyak val rmse = ', val_rmse)
        pass


    #print('global seed = ', seed)
    exp_dict[str(seed)]=epoch_dict
    # For parsing results from file
    print('best_val_score', float(round(best_val_score,3)), 'best_epoch', best_epoch)

    sess.close()

X_plot = np.arange(NB_EPOCH)
y_tr = []
y_v = []
for key in exp_dict.keys():
    y_tr.append(exp_dict[key]['t_er'])
    y_v.append(exp_dict[key]['v_er'])
y_tr=np.array(y_tr)
print(y_tr.shape)
y_tr = np.mean(y_tr,axis=0)
print(y_tr.shape)
y_v = np.mean(np.array(y_v),axis=0)

plt.plot(X_plot, y_tr, label = "Train Error")
plt.plot(X_plot, y_v, label = "Val Error")
plt.legend()
plt.show()
