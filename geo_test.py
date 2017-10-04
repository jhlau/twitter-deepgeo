"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Mar 17
"""

import argparse
import sys
import codecs
import random
import operator
import os
import cPickle
import math
import scipy.io as sio
import time as tm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
from collections import namedtuple
from functools import partial
from geo_model import TGP
from util import *

#parser arguments
desc = "Given trained model, perform various test inferences"
parser = argparse.ArgumentParser(description=desc)

###################
#optional argument#
###################
parser.add_argument("-m", "--model_dir", required=True, help="directory of the saved model")
parser.add_argument("-d", "--input_doc", help="input file containing the test documents")
parser.add_argument("-l", "--input_label", help="input file containing the test labels")
parser.add_argument("--predict", help="classify test instances and compute accuracy", action="store_true")
parser.add_argument("--save_rep", help="save representation (thresholded and converted to binary) of test instances")
parser.add_argument("--save_label", help="save label of test instances")
parser.add_argument("--save_mat", help="save representation (floats) and label of test instances in MAT format")
parser.add_argument("--print_attn", help="print attention on text span", action="store_true")
parser.add_argument("--print_time", help="print time, offset and usertime distribution for popular locations", \
    action="store_true")
args = parser.parse_args()

###########
#functions#
###########
def get_mu_sigma(model, var_name):
    with tf.variable_scope("model"):
        m, s = sess.run(model.get_mu_sigma(var_name))
    return m, s

def run_epoch(data, label, classes_rev, models, cf):
    reps, labels = [], []
    class_count = defaultdict(int) #count for each class
    time_dist = [defaultdict(partial(np.ndarray, cf.time_size)), defaultdict(partial(np.ndarray, cf.offset_size)), \
        defaultdict(partial(np.ndarray, cf.usertime_size))] #dist for each class (time, offset and usertime features)
    start_time = tm.time()
    costs, accs = 0.0, []
    num_batches = int(math.ceil(float(len(data))/cf.batch_size))
    batch_ids = range(num_batches)

    #create text length to bucket id map
    lenxbucket, prev_b = {}, -1
    for bi, b in enumerate(cf.bucket_sizes):
        for i in xrange(prev_b+1, b+1):
            lenxbucket[i] = (bi, b)
        prev_b = b

    #generate results to call to models in different buckets
    res = []
    for bi, b in enumerate(cf.bucket_sizes):
        res.append([models[bi].cost, models[bi].probs, models[bi].rep, models[bi].text_attn, \
            models[bi].hidden_time if cf.time_size > 0 else tf.no_op(), \
            models[bi].hidden_offset if cf.offset_size > 0 else tf.no_op(), \
            models[bi].hidden_usertime if cf.usertime_size > 0 else tf.no_op()])
    
    #iterate through the batches
    for ni, i in enumerate(batch_ids):
        x, y, time, day, offset, timezone, loc, desc, name, usertime, noise, num_examples, b = \
            get_batch(data, i, lenxbucket, False, cf)

        cost, prob, r, attn, time, offset, usertime = \
            sess.run(res[b], {models[b].x:x, models[b].y:y, models[b].time:time, models[b].day:day, \
            models[b].offset:offset, models[b].timezone:timezone, models[b].loc:loc, models[b].desc:desc, \
            models[b].name:name, models[b].usertime:usertime, models[b].noise:noise})

        costs += cost
        pred = np.argmax(prob, axis=1)
        accs.extend(pred[:num_examples] == y[:num_examples])
        reps.extend(list(np.reshape(r[:num_examples], [-1])))
        labels.extend(y[:num_examples])

        #collect counts and dist for a time features
        for j in range(num_examples):
            class_count[y[j]] += 1
            time_dist[0][y[j]] += time[j]
            time_dist[1][y[j]] += offset[j]
            time_dist[2][y[j]] += usertime[j]

        if args.print_attn:
            #print character span attention for each instance
            pos = np.fliplr(np.argsort(attn))
            for j in range(attn.shape[0]):
                inst_id = i*cf.batch_size+j

                if inst_id < len(data):
                    text = data[inst_id]["text"]
                    print "\nInstance ID     =", inst_id
                    print "Text            =", text
                    print "True Label      =", label[data[inst_id]["id_str"]]
                    print "Predicted Label =", classes_rev[pred[j]]
                    for k in range(20):
                        print "\t%.5f : [%3d] %s" % \
                            (attn[j][pos[j][k]], pos[j][k], text[pos[j][k]:pos[j][k]+cf.text_pool_window])
                else:
                    break
        else:
            #print progress
            if ((ni % 10) == 0) or (ni == num_batches-1):
                sys.stdout.write("TEST ")
                sys.stdout.write("%d/%d: avg loss = %.3f; avg acc = %.3f; inst/sec = %.1f" % \
                    (ni+1, num_batches, costs/(ni+1), np.mean(accs), float((ni+1)*cf.batch_size)/(tm.time()-start_time)))
                if ni == (num_batches-1):
                    sys.stdout.write("\n")
                else:
                    sys.stdout.write("\r")
                sys.stdout.flush()

    #print time, offset and user time distribution 
    if args.print_time:
        top_n = 10
        cc = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
        if cf.time_size > 0:
            time_mu, time_sigma = get_mu_sigma(models[0], "time")
        if cf.offset_size > 0:
            offset_mu, offset_sigma = get_mu_sigma(models[0], "offset")
        if cf.usertime_size > 0:
            usertime_mu, usertime_sigma = get_mu_sigma(models[0], "usertime")
        print "\nTime, offset and usertime distribution for top-N Cities:"
        for k, v in cc:
            print "\n", v, ":", classes_rev[k]
            if cf.time_size > 0:
                time_ids = np.argsort(time_dist[0][k])[::-1]
                print "\tTime distributions"
                for t in time_ids[:top_n]:
                    print "\t\t%.2f = %02d:%02d" % (time_dist[0][k][t]/v, \
                        int((time_mu[t]*1440)/60), int((time_mu[t]*1440)%60))

                x = np.linspace(-0.0, 1.0, 10000)
                time_sigma = np.absolute(time_sigma)
                mpl.rc('font', size=24, family="Times New Roman")
                for i in range(0, time_mu.shape[0]):
                    t_weight = time_dist[0][k][i]/v*4.0
                    if t_weight > 1.0:
                        t_weight = 1.0
                    if t_weight < 0.3:
                        t_weight = 0.0
                    plt.plot(x,mlab.normpdf(x, time_mu[i], time_sigma[i]), "b", alpha=t_weight)
                labels = ("00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00")
                plt.ylim(-0.1, 25)
                plt.xticks(tuple([item * (1.0/6) for item in range(7)]), labels)
                #plt.xlabel("Time (UTC)")
                plt.title(classes_rev[k])
                plt.show()
                plt.clf()

            if cf.offset_size > 0:
                offset_ids = np.argsort(time_dist[1][k])[::-1]
                print "\tOffset distributions"
                for t in offset_ids[:top_n]:
                    print "\t\t%.2f = %.2f" % (time_dist[1][k][t]/v, (offset_mu[t]*26.0)-12.0)
            if cf.usertime_size > 0:
                usertime_ids = np.argsort(time_dist[2][k])[::-1]
                print "\tUser time distributions"
                for t in usertime_ids[:top_n]:
                    print "\t\t%.2f = %02d:%02d" % (time_dist[2][k][t]/v, \
                        int((usertime_mu[t]*1440)/60), int((usertime_mu[t]*1440)%60))
            
            

    return np.array(reps), np.mean(accs), np.array(labels)


######
#main#
######
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

#load the vocabulary
vocabxid, tzxid, locxid, descxid, namexid, classes = cPickle.load(open(os.path.join(args.model_dir, "feature_ids.pickle")))
classes_rev = [ item[0] for item in sorted(classes.items(), key=operator.itemgetter(1)) ]

#load config
cf_dict = cPickle.load(open(os.path.join(args.model_dir, "config.pickle")))
if "text_pool_stride" in cf_dict: #fix for older models
    cf_dict["text_pool_window"] = cf_dict["text_pool_stride"]
    del cf_dict["text_pool_stride"]
ModelConfig = namedtuple("ModelConfig", " ".join(cf_dict.keys()))
cf = ModelConfig(**cf_dict)

#set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)

#load raw test data and labels
print "Loading test labels..."
test_label = load_label(args.input_label, cf)
print "Loading test data..."
test_data = load_data(args.input_doc, test_label, False, cf)

#clean text data
print "Converting text to ids..."
test_len_x, test_miss_y, test_len_loc, test_len_desc, test_len_name = clean_data(test_data, test_label, \
    vocabxid, tzxid, locxid, descxid, namexid, classes, cf)

print "\nStatistics:"
print "Number of test instances =", len(test_data)
print ("Test:\n\tmean/max text len = %.2f/%d;" + \
    "\n\tmean/max location len = %.2f/%d;" + \
    "\n\tmean/max description len = %.2f/%d;" + \
    "\n\tmean/max name len = %.2f/%d;" + \
    "\n\tno. instances with missing classes = %d") % \
    (np.mean(test_len_x), max(test_len_x), np.mean(test_len_loc), max(test_len_loc), \
    np.mean(test_len_desc), max(test_len_desc), np.mean(test_len_name), max(test_len_name), test_miss_y)

#initialise and load model parameters
with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(cf.seed)
    initializer = tf.contrib.layers.xavier_initializer(seed=cf.seed)
    mtests = []
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtests.append(TGP(is_training=False, vocab_size=len(vocabxid), num_steps=cf.bucket_sizes[0], \
            num_classes=len(classes), num_timezones=len(tzxid), loc_vsize=len(locxid), \
            desc_vsize=len(descxid), name_vsize=len(namexid), config=cf))
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        if len(cf.bucket_sizes) > 1:
            for b in cf.bucket_sizes[1:]:
                mtests.append(TGP(is_training=False, vocab_size=len(vocabxid), num_steps=b, \
                    num_classes=len(classes), num_timezones=len(tzxid), loc_vsize=len(locxid), \
                    desc_vsize=len(descxid), name_vsize=len(namexid),  config=cf))


    #load tensorflow model
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(args.model_dir, "model.ckpt"))

    #predict test instances geolocation
    if args.predict or args.save_rep or args.save_label or args.print_attn or args.save_mat or args.print_time:
        reps, acc, labels = run_epoch(test_data, test_label, classes_rev, mtests, cf)
        if args.predict:
            print "\nTest accuracy =", acc
        if args.save_rep:
            x = reps.reshape((-1, cf.rep_hidden_size))>0
            #x = reps.reshape((-1, cf.rep_hidden_size))
            np.save(open(args.save_rep, "w"), x)
        if args.save_label:
            np.save(open(args.save_label, "w"), labels)
        if args.save_mat:
            x = reps.reshape((-1, cf.rep_hidden_size))
            sio.savemat(args.save_mat, {"dataset": {"x": x, "y": labels.reshape((-1, 1))}})
