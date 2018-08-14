import pandas as pd
import numpy as np

"""
Class that performs the estimation (and validation) for the model
and some auxiliary functions
"""


class TrainExp:
    def __init__(self, data, sm, settings):
        self.data = data
        self.sm = sm
        self.iterations = settings['iterations']
        self.warmup = settings['warmup']
        self.chains = settings['chains']
        self.n_jobs = settings['n_jobs']    

        self.train_data, self.validation_data = split_to_train_validation(self.data)

        self.fit = None

    def fit_model(self):
        T = len(self.train_data.index.get_level_values('datetime').unique())    
        N = len(self.train_data.index.get_level_values('id').unique())    
        train_data = self.train_data
        train_data.index = self.train_data.index.droplevel(level=['month', 'weekday']) 
        E = train_data.unstack()
        stan_data = {'T': T, 'N': N, 'E': E}
        self.fit = self.sm.sampling(data=stan_data, iter=self.iterations, warmup=self.warmup, chains=self.chains, n_jobs = self.n_jobs, seed=1)
        print(self.fit)

    def get_cust_ids(self):
        customer_ids = self.train_data.index.get_level_values('id').unique()
        return customer_ids  

    def get_local_samples(self):
        samples = self.fit.extract(pars=['lambda'], permuted=True)
        return samples

    def print_test_summary(self):
        """
        Here we make predictions based on parameter samples. Parameter distributions are reduced to point estimates
        by taking the average. This gives us an exponential distribution for each customer. The mean of such a distribution
        is the inverse of the parameter lambda again, hence we can use this value immediately. However, beause
        the mean of a nonlinear function does not in general equal the nonlinear function of the mean, we have to first
        take the inverse of all the samples, then take the average.
        """

        samples = self.get_local_samples()['lambda']
        inverse_samples = 1 / samples
        pred = np.mean(inverse_samples, axis=0) 
        pred_s = pd.Series(pred, index=self.train_data.index.get_level_values('id').unique()) 
        self.train_data = self.train_data.groupby(level=[0]).apply(add_pred_per_id, pred=pred_s)
        self.train_data['error'] = np.fabs(self.train_data['usage'] - self.train_data['pred'])
        print(self.train_data['error'].describe())    


def add_pred_per_id(df, pred):
    id = list(df.index.get_level_values('id').unique())[0]
    pred_id = pred.loc[id]
    df['pred'] = pred_id
    return df

def split_to_train_validation(df, train_frac=1):
    datetime_values = df.index.get_level_values('datetime').unique()
    n_train = int(np.floor(train_frac * len(datetime_values)))
    idx = pd.IndexSlice
    df.sort_index(inplace=True)
    train_df = df.loc[idx[:, slice(datetime_values[0], datetime_values[n_train - 1]), :, :], :]
    validation_df = df.loc[idx[:, slice(datetime_values[n_train - 1], datetime_values[len(datetime_values) - 1]), :, :], :]
    return train_df, validation_df       