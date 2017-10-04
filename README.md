# Requirements
- python2.7
- tensorflow 0.8-0.12
- ujson
- numpy
- scipy (for retrieval_map.py, geo_test.py)
- matplotlib (for geo_test.py)

# Data
- Validation (development) and test tweet data are provided (under data/)
- For training tweets, use the training downloader script from [WNUT Workshop 2016](http://noisy-text.github.io/2016/geo-shared-task.html). The script will fetch the training tweet metadata from the official API

# Pre-trained Models
- Pre-trained models can be downloaded [here](https://ibm.box.com/s/q3do4wzas31dpg9121ljlati093gn2cr) (1.6GB)
- There are a total of 12 models:
  - deepgeo_RXXX: R=XXX, sigma=0.0, alpha=0.0
  - deepgeo_RXXX_noise: R=XXX, sigma=0.1, alpha=0.0
  - deepgeo_RXXX_loss: R=XXX, sigma=0.0, alpha=0.1
- These are the models presented in Table 6 in the paper. R is the dimension of the representation, sigma is the Gaussian noise standard deviation, and alpha is the scaling factor the additional loss term l

# Train
- `python geo_train.py`
- Configurations are all defined in config.py
- The default values are the optimal hyper-parameter settings used in the paper

# Test
```
usage: geo_test.py [-h] -m MODEL_DIR [-d INPUT_DOC] [-l INPUT_LABEL]
                   [--predict] [--save_rep SAVE_REP] [--save_label SAVE_LABEL]
                   [--save_mat SAVE_MAT] [--print_attn] [--print_time]

Given trained model, perform various test inferences

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model_dir MODEL_DIR
                        directory of the saved model
  -d INPUT_DOC, --input_doc INPUT_DOC
                        input file containing the test documents
  -l INPUT_LABEL, --input_label INPUT_LABEL
                        input file containing the test labels
  --predict             classify test instances and compute accuracy
  --save_rep SAVE_REP   save representation (thresholded and converted to
                        binary) of test instances
  --save_label SAVE_LABEL
                        save label of test instances
  --save_mat SAVE_MAT   save representation (floats) and label of test
                        instances in MAT format
  --print_attn          print attention on text span
  --print_time          print time, offset and usertime distribution for
                        popular locations
```

#### Example: Compute test data accuracy
`python geo_test.py -m MODEL_DIR -d data/test/data.tweet.json -l data/test/label.tweet.json --predict`

#### Example: Print character attention in test data
`python geo_test.py -m MODEL_DIR -d data/test/data.tweet.json -l data/test/label.tweet.json --print_attn`

# Compute MAP Performance
- Example script is given in: compute_map.sh
- The idea is to first generate binary code representation for train and test data, and then use retrieval_map to compute hamming distance and MAP
- In the example script, we use the validation data as the train data, as it is much smaller

# Publication

Lau, Jey Han, Lianhua Chi, Khoi-Nguyen Tran and Trevor Cohn (to appear). End-to-end Network for Twitter Geolocation Prediction and Hashing. In Proceedings of the 8th International Joint Conference on Natural Language Processing (IJCNLP 2017), Taipei, Taiwan.
