// very basic hierarchical estimation model
data {
    int<lower=1> N; // number of customers
    int<lower=1> T; // number of time points
    matrix<lower=0>[N,T] E; // energy usage per customer per time point
}

parameters {
    real mu; // first global parameter, mean
    real<lower=0> sigma; // second global parameter, standard deviation
    vector[N] phi_tilde; // auxiliary transformed local parameters
}

transformed parameters {
    vector[N] phi; // transformed local parameters
    vector<lower=0>[N] lambda; // local parameters
    
    phi = mu + sigma * phi_tilde;
    lambda = exp(phi);
}

model {
    // sampling efficiency is improved by transforming parameters to take real values
    // and by sampling local parameters step-wise	
    mu ~ normal(0, 1);
    sigma ~ normal(0, 1);
    phi_tilde ~ normal(0, 1); 

    for (i in 1:N) {
        for (j in 1:T) {
            E[i,j] ~ exponential(lambda[i]);
        }
    }
}
