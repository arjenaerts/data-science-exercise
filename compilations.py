import os
import pickle
import pystan

"""
Script that compiles models in advance in order to save time during execution
"""

dir_name = os.getcwd() + '/stan_models/'

sm = pystan.StanModel(file=dir_name + 'usage_exp.stan', verbose=True)
with open(dir_name + 'usage_exp_model.pkl', 'wb') as f:
    pickle.dump(sm, f)

# sm = pystan.StanModel(file=dir_name + 'usage_exp_dummy.stan')
# with open(dir_name + 'usage_exp_dummy_model.pkl', 'wb') as f:
#     pickle.dump(sm, f)

# sm = pystan.StanModel(file=dir_name + 'usage_exp_dummy_markov.stan')
# with open(dir_name + 'usage_exp_dummy_markov_model.pkl', 'wb') as f:
#     pickle.dump(sm, f)        