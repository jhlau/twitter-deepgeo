"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Nov 16
"""

import argparse
import sys
import codecs
import imp
import operator
import random
import os
import cPickle
import time as tm
import tensorflow as tf
import numpy as np
from geo_model import TGP
from util import *

#parser arguments
desc = "Given train/valid json tweets, train neural network to predict tweet locations"
parser = argparse.ArgumentParser(description=desc)

###################
#optional argument#
###################
#parser.add_argument("-v", "--verbosity", help="")
parser.add_argument("-c", "--config", help="path of config file")
args = parser.parse_args()

#load config
if args.config:
    print "Loading config from:", args.config
    cf = imp.load_source('config', args.config)
else:
    print "Loading config from default directory"
    import config as cf

###########
#functions#
###########
def run_epoch(data, models, is_training):
    reps = []
    start_time = tm.time()
    costs, accs = 0.0, []
    num_batches = int(len(data)/cf.batch_size)
    batch_ids = range(num_batches)
    random.shuffle(batch_ids)

    #create text length to bucket id map
    lenxbucket, prev_b = {}, -1
    for bi, b in enumerate(cf.bucket_sizes):
        for i in xrange(prev_b+1, b+1):
            lenxbucket[i] = (bi, b)
        prev_b = b
    
    #generate results to call to models in different buckets
    res = []
    for bi, b in enumerate(cf.bucket_sizes):
        res.append([models[bi].cost, models[bi].probs, models[bi].rep, models[bi].train_op \
            if is_training else tf.no_op()])

    for ni, i in enumerate(batch_ids):
        x, y, time, day, offset, timezone, loc, desc, name, usertime, noise, _, b = \
            get_batch(data, i, lenxbucket, is_training, cf)
        cost, prob, dn , _ = sess.run(res[b], {models[b].x:x, models[b].y:y, models[b].time:time, models[b].day:day, \
            models[b].offset:offset, models[b].timezone:timezone, models[b].loc:loc, models[b].desc:desc, \
            models[b].name:name, models[b].usertime:usertime, models[b].noise:noise})
        costs += cost
        pred = np.argmax(prob, axis=1)
        accs.extend(pred == y)
        if not is_training:
            reps.extend(list(np.reshape(dn, [-1])))

        #print some training statistics
        if (((ni % 10) == 0) and cf.verbose) or (ni == num_batches-1):
            if is_training:
                sys.stdout.write("TRAIN ")
            else:
                sys.stdout.write("VALID ")
            sys.stdout.write("%d/%d: avg loss = %.3f; avg acc = %.3f; inst/sec = %.1f" % \
                (ni+1, num_batches, costs/(ni+1), np.mean(accs), float((ni+1)*cf.batch_size)/(tm.time()-start_time)))
            if ni == (num_batches-1):
                sys.stdout.write("\n")
            else:
                sys.stdout.write("\r")
            sys.stdout.flush()

    return np.array(reps), np.mean(accs)


######
#main#
######
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
#set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)

#load raw train and valid data and labels
print "Loading train and valid labels..."
train_label = load_label(cf.train_label, cf)
valid_label = load_label(cf.valid_label, cf)

print "Loading train and valid data..."
train_data = load_data(cf.train_data, train_label, True, cf)
valid_data = load_data(cf.valid_data, valid_label, False, cf)

#collect vocab and classes
print "Collecting text vocab..."
vocabxid, idxvocab, _ = get_vocab(train_data, "text", "char", cf.word_minfreq)
print "Collecting time zone vocab..."
tzxid, _, _ = get_vocab(train_data, "timezone", "word", 0)
print "Collecting location vocab..."
locxid, _, _ = get_vocab(train_data, "location", "char", cf.word_minfreq)
print "Collecting description vocab..."
descxid, _, _ = get_vocab(train_data, "description", "char", cf.word_minfreq)
print "Collecting name vocab..."
namexid, _, _ = get_vocab(train_data, "name", "char", cf.word_minfreq)
print "Collecting class labels..."
classes = get_classes(train_data, train_label)

#clean text data
print "Converting text to ids..."
train_len_x, train_miss_y, train_len_loc, train_len_desc, train_len_name = clean_data(train_data, train_label, \
    vocabxid, tzxid, locxid, descxid, namexid, classes, cf)
valid_len_x, valid_miss_y, valid_len_loc, valid_len_desc, valid_len_name = clean_data(valid_data, valid_label, \
    vocabxid, tzxid, locxid, descxid, namexid, classes, cf)

#Sorting data based on length
print "Sorting data based on tweet length..."
train_data = sorted(train_data, key=lambda item: len(item["x"]))
valid_data = sorted(valid_data, key=lambda item: len(item["x"]))

print "\nStatistics:"
print "Number of train instances =", len(train_data)
print "Number of valid instances =", len(valid_data)
print "Text vocab size =", len(vocabxid)
print "Location vocab size =", len(locxid)
print "Description vocab size =", len(descxid)
print "Name vocab size =", len(namexid)
print "Class size =", len(classes)
print "No. of timezones =", len(tzxid)
print ("Train:\n\tmean/max text len = %.2f/%d;" + \
    "\n\tmean/max location len = %.2f/%d;" + \
    "\n\tmean/max description len = %.2f/%d;" + \
    "\n\tmean/max name len = %.2f/%d;" + \
    "\n\tno. instances with missing classes = %d") % \
    (np.mean(train_len_x), max(train_len_x), np.mean(train_len_loc), max(train_len_loc), \
    np.mean(train_len_desc), max(train_len_desc), np.mean(train_len_name), max(train_len_name), train_miss_y)
print ("Valid:\n\tmean/max text len = %.2f/%d;" + \
    "\n\tmean/max location len = %.2f/%d;" + \
    "\n\tmean/max description len = %.2f/%d;" + \
    "\n\tmean/max name len = %.2f/%d;" + \
    "\n\tno. instances with missing classes = %d") % \
    (np.mean(valid_len_x), max(valid_len_x), np.mean(valid_len_loc), max(valid_len_loc), \
    np.mean(valid_len_desc), max(valid_len_desc), np.mean(valid_len_name), max(valid_len_name), valid_miss_y)

#train model
with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(cf.seed)
    initializer = tf.contrib.layers.xavier_initializer()
    mtrains, mvalids = [], []
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtrains.append(TGP(is_training=True, vocab_size=len(idxvocab), num_steps=cf.bucket_sizes[0], \
            num_classes=len(classes), num_timezones=len(tzxid), loc_vsize=len(locxid), \
            desc_vsize=len(descxid), name_vsize=len(namexid), config=cf))
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        if len(cf.bucket_sizes) > 1:
            for b in cf.bucket_sizes[1:]:
                mtrains.append(TGP(is_training=True, vocab_size=len(idxvocab), num_steps=b, \
                    num_classes=len(classes), num_timezones=len(tzxid), loc_vsize=len(locxid), \
                    desc_vsize=len(descxid), name_vsize=len(namexid),  config=cf))
        for b in cf.bucket_sizes:
            mvalids.append(TGP(is_training=False, vocab_size=len(idxvocab), num_steps=b, \
                num_classes=len(classes), num_timezones=len(tzxid), loc_vsize=len(locxid), \
                desc_vsize=len(descxid), name_vsize=len(namexid), config=cf))

    tf.initialize_all_variables().run()

    #save model every epoch
    if cf.save_model:
        if not os.path.exists(os.path.join(cf.output_dir, cf.output_prefix)):
            os.makedirs(os.path.join(cf.output_dir, cf.output_prefix))
        #create saver object to save model
        saver = tf.train.Saver()

    #train model
    reps = None
    prev_acc = 0.0
    for i in xrange(cf.epoch_size):
        print "\nEpoch =", i
        #run a train epoch
        run_epoch(train_data, mtrains, True)
        #run a valid epoch
        reps, acc = run_epoch(valid_data, mvalids, False)

        if cf.save_model:
            if acc > prev_acc:
                saver.save(sess, os.path.join(cf.output_dir, cf.output_prefix, "model.ckpt"))
                prev_acc = acc
            else:
                saver.restore(sess, os.path.join(cf.output_dir, cf.output_prefix, "model.ckpt"))
                print "\tNew valid performance > prev valid performance: restoring previous parameters..."

    #save time parameters
    if cf.save_model:
        if not os.path.exists(os.path.join(cf.output_dir, cf.output_prefix)):
            os.makedirs(os.path.join(cf.output_dir, cf.output_prefix))
        np.save(open(os.path.join(cf.output_dir, cf.output_prefix, "rep.npy"), "w"), reps)

        #feature ID information
        cPickle.dump((vocabxid, tzxid, locxid, descxid, namexid, classes), \
            open(os.path.join(cf.output_dir, cf.output_prefix, "feature_ids.pickle"), "w"))

        #create a dictionary object for config
        cf_dict = {}
        for k,v in vars(cf).items():
            if not k.startswith("__"):
                cf_dict[k] = v
        cPickle.dump(cf_dict, open(os.path.join(cf.output_dir, cf.output_prefix, "config.pickle"), "w"))
