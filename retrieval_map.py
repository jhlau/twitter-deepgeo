"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Mar 16
"""

import argparse
import sys
import os
import operator
import numpy as np
import scipy.io as sio
from collections import defaultdict
from scipy.spatial.distance import cdist, hamming

#parser arguments
desc = "Compute mean average precision of retrieval (based on binary hashcode and Hamming distance)"
parser = argparse.ArgumentParser(description=desc)

###################
#optional argument#
###################
parser.add_argument("-i", "--input_dir", help="input directory containing training and test data", required=True)
parser.add_argument("--train_code", help="file containing train hashcode (mat or npy)")
parser.add_argument("--test_code", help="file containing test hashcode (mat or npy)")

args = parser.parse_args()

#parameters
debug = False
min_n = 1
distance = "hamming" #hamming or cosine

train_label_fname = "valid_label.npy"
test_label_fname = "test_label.npy"
if args.train_code:
    train_rep_fname = args.train_code
else:
    #train_rep_fname = "compressedValid.mat"
    train_rep_fname = "valid_rep.npy"

if args.test_code:
    test_rep_fname = args.test_code
else:
    #test_rep_fname = "compressedTest.mat"
    test_rep_fname = "test_rep.npy"

###########
#functions#
###########
def get_key(x):
    for k in x.keys():
        if not k.startswith("__"):
            return k

######
#main#
######

#load the data
print "Loading train and test data..."
train_label = np.load(open(os.path.join(args.input_dir, train_label_fname)))
test_label = np.load(open(os.path.join(args.input_dir, test_label_fname)))

if train_rep_fname.endswith(".npy"): 
    train_hash = np.load(open(os.path.join(args.input_dir, train_rep_fname)))
else:
    data =  sio.loadmat(os.path.join(args.input_dir, train_rep_fname))
    key = get_key(data)
    train_hash = np.array(data[key], dtype=bool)
if test_rep_fname.endswith(".npy"):
    test_hash = np.load(open(os.path.join(args.input_dir, test_rep_fname)))
else:
    data =  sio.loadmat(os.path.join(args.input_dir, test_rep_fname))
    key = get_key(data)
    test_hash = np.array(data[key], dtype=bool)

print "Number train instances =", train_hash.shape[0]
print "Number test instances =", test_hash.shape[0]

#filter away test instances that have less than N train neighbours
print "\nFiltering away test instances that have less than %d neighbours in train..." % min_n
label_count = defaultdict(int)
test_hash_f, test_label_f = [], []
for l in train_label:
    label_count[l] += 1
for li, l in enumerate(test_label):
    if l in label_count and label_count[l] >= min_n:
        test_hash_f.append(test_hash[li])
        test_label_f.append(l)
test_hash_f = np.array(test_hash_f)
test_label_f = np.array(test_label_f)

print "Number of filtered test instances =", test_hash_f.shape[0]

#compute hamming distance for all pairs between train and test instances
print "\nComputing hamming distance..."
if distance == "hamming":
    h = cdist(test_hash_f, train_hash, "hamming")
else:
    h = cdist(test_hash_f, train_hash, "cosine")

#compute mean average precision
print "\nComputing mean average precision..."
avps = []
hidden_size = test_hash_f.shape[1]
for y in range(h.shape[0]):
    y_label = test_label_f[y]
    xs = np.argsort(h[y])
    num_neighbours = label_count[y_label]
    avp, true_positive = 0.0, 0
    if debug:
        print "\ntest instance =", y
        #print "test instance hash =", np.array(test_hash_f[y], dtype=int)
        print "test instance label =", y_label
        print "number of neighbours in train =", num_neighbours
    for xi, x in enumerate(xs):
        if debug:
            print "\n\t", xi, "train instance =", x
            #print "\t\ttrain instance hash =", np.array(train_hash[x], dtype=int)
            print "\t\ttrain instance label =", train_label[x]
            print "\t\thamming distance =", hamming(train_hash[x], test_hash_f[y])
        if y_label == train_label[x]:
            true_positive += 1
            avp += float(true_positive) / (xi+1)
            if debug:
                print "\t\t\tHIT!", true_positive, avp

    avp = avp / label_count[y_label]
    avps.append(avp)
    if debug:
        print "\n\taverage precision =", avps[-1]

print "Mean average precision =", np.mean(avps) #, np.std(avps), np.median(avps), np.max(avps), np.min(avps)
