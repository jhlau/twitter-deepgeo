#!/bin/bash

#parameters
model_dir="model/deepgeo_R100"
valid_data="data/valid/data.tweet.json"
valid_label="data/valid/label.tweet.json"
test_data="data/test/data.tweet.json"
test_label="data/test/label.tweet.json"

echo "processing $model_dir..."
#first generate binary code for valid
python geo_test.py -m $model_dir -d $valid_data -l $valid_label --save_rep $model_dir/valid_rep.npy \
    --save_label $model_dir/valid_label.npy 2>/dev/null
#generate binary code for test
python geo_test.py -m $model_dir -d $test_data -l $test_label --save_rep $model_dir/test_rep.npy \
    --save_label $model_dir/test_label.npy 2>/dev/null
#compute MAP performance (test against valid)
python retrieval_map.py -i $model_dir
