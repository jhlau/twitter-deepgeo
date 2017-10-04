# Requirements
- python2.7
- tensorflow 0.8-0.12
- ujson
- numpy
- scipy (for retrieval_map.py, geo_test.py)
- matplotlib (for geo_test.py)

# Data
- Validation (development) and test tweet data are provided (under data/)
- For training tweets, use the training downloader script from [WNUT Workshop 2016](http://noisy-text.github.io/2016/geo-shared-task.html). The script will fetch the training tweet metadata from the official API.

# Pre-trained Models
- Pre-trained models can be downloaded [here] (https://ibm.box.com/s/q3do4wzas31dpg9121ljlati093gn2cr) (1.6GB)
- There are a total of 12 models:
  - deepgeo_RXXX: R=XXX, sigma=0.0, alpha=0.0
  - deepgeo_RXXX_noise: R=XXX, sigma=0.1, alpha=0.0
  - deepgeo_RXXX_loss: R=XXX, sigma=0.0, alpha=0.1
- These are the models presented in Table 6 in the paper. R is the dimension of the representation, sigma is the Gaussian noise standard deviation, and alpha is the scaling factor the additional loss term l.
