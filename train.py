import pandas as pd
import numpy as np
import pickle
import engine as en

"""
Script that reads the data, transforms / cleans the data, splits the data into train and validation data,
trains and validates three models, selects the best model and outputs the estimation results (not yet complete)

First COMPILE models on a new system using compilations.py
"""

data = pd.read_csv('usage_train.csv')

# tidy data
data['datetime'] = pd.to_datetime(data['datetime'])
data['month'] = data['datetime'].map(lambda x: x.month)
data['weekday'] = data['datetime'].dt.dayofweek 
data = data.set_index(['id', 'datetime', 'month', 'weekday'])

# est_settings for Stan
est_settings = {'iterations': 150, 'warmup': 100, 'chains': 4, 'n_jobs': 1} 

# load compiled Stan model
with open('stan_models/' + 'usage_exp_model.pkl', 'rb') as f:
    sm = pickle.load(f)

full_estimation = en.TrainExp(data, sm, est_settings)
full_estimation.fit_model()
full_estimation.print_test_summary()
local_samples = full_estimation.get_local_samples()
cust_ids = full_estimation.get_cust_ids()

# save estimation result
with open('local_samples.pkl', 'wb') as f:
    pickle.dump(local_samples, f)

# save cust_ids
with open('cust_ids.pkl', 'wb') as f:
    pickle.dump(cust_ids, f)

#########################################################################
# REMAINDER

# three models:
# -use separate exponential distribution (maximum entropy) for each customer, no dummies 
# -use separate exponential distribution (maximum entropy) for each customer, with individual-specific dummies in the lambda parameters
# -include 2-state Markov model by separating usage into low-usage and high-usage, and combine with simple distributions
# for both regimes (say, exponential); the pars of the exponential distributions should be a function of dummies  

# make two naive predictions with (a) total average usage and (b) individual usages

# for each model do:
# -create Stan script
# -split usage into train_data and validation_data (validation period) 
#  should be around 10 percent of time-period
# -transform usage data into stan_data
# -when incorporating the weekly and monthly data, make dummies for Stan
# -when incorporating two states, create extra column with states for Stan, using 25th quantile threshold
# -run model
# -compute train error by making predicting for each row in train_data and computing MAE
# -compute validation error by making prediction for each row in validation_data and computing MAE
# -put all this in a single class 
# -predictions based on average