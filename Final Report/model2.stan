
data {
  int<lower=0> N; //Number of exonorees
  int<lower=0> S; //Number of states;
  int<lower=0> RC; //Number of race-crime groups
  int state[N]; //Indicator for which state group 
  int race_crime[N]; //Indicator for which race-crime group 
  int age[N]; //Age of exonorees
  int gender[N]; //Indicator of gender where 1 represents male, 0 represents female
  vector<lower=0>[N] t; //Time-to-exoneration 
}


parameters {
  vector[S] alpha_state;
  vector[RC] alpha_RC;
  vector[3] beta; 
  real<lower=0> sigma_state;
  real<lower=0> sigma_RC;
  real<lower=0> sigma;
}

model {
  vector[N] p;
  
  //Log-likelihood
  for (n in 1:N) {
    target += normal_lpdf(t[n] | beta[1] + beta[2]*age[n] + beta[3]*gender[n] + alpha_RC[race_crime[n]] + alpha_state[state[n]], sigma);
  }
  // Priors
  target +=  normal_lpdf(alpha_RC | 0, sigma_RC);
  target +=  normal_lpdf(alpha_state | 0, sigma_state);
  target += normal_lpdf(beta | 0, 1);
  target += normal_lpdf(sigma_RC | 0, 1);
  target += normal_lpdf(sigma_state | 0, 1);
  target += normal_lpdf(sigma | 0, 1);
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] log_weight_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real mu = beta[1] + beta[2]*age[n] + beta[3]*gender[n] + alpha_RC[race_crime[n]] + alpha_state[state[n]];
    log_lik[n] = normal_lpdf(t[n] | mu, sigma);
    log_weight_rep[n] = normal_rng(mu, sigma);
  }
}