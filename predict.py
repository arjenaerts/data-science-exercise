import pandas as pd
import numpy as np
import pickle
import engine as en

"""
Script that makes predictions for a set of customer-time pairs using the model output from train.py
"""

# load customer-time pairs to be scored
data = pd.read_csv('usage_test.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index(['id', 'datetime'])

# load pickled files with samples 
with open('local_samples.pkl', 'rb') as f:
    local_samples = pickle.load(f)

# load pickled files with cust_ids (this is required because the sequence of cust_ids matters, since Stan has no indexing)
with open('cust_ids.pkl', 'rb') as f:
    cust_ids = pickle.load(f)

# make predictions based on samples
samples = local_samples['lambda']
inverse_samples = 1 / samples
pred = np.mean(inverse_samples, axis=0) 
pred_s = pd.Series(pred, index=cust_ids)
data = data.groupby(level=[0]).apply(en.add_pred_per_id, pred=pred_s)

# write data to csv file
data.to_csv('usage_test_predictions.csv')