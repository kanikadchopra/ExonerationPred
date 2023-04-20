
data {
  int<lower=0> N; //Number of exonorees
  int<lower=0> S; //Number of states;
  int<lower=0> R; //Number of race groups
  int<lower=0> C; //Number of crime groups 
  int state[N]; //Indicator for which state group 
  int race[N]; //Indicator for which race group 
  int crime[N]; //Indicator for which crime wrongfully convicted of
  int age[N]; //Age of exonorees
  int gender[N]; //Indicator of gender where 1 represents male, 0 represents female
  vector<lower=0>[N] t; //Time-to-exoneration 
}

parameters {
  vector[S] alpha_state;
  vector[R] alpha_race;
  vector[C] alpha_crime;
  vector[3] beta; 
  real<lower=0> sigma_state;
  real<lower=0> sigma_race;
  real<lower=0> sigma_crime;
  real<lower=0> sigma;
}

model {
  vector[N] p;
  
  //Log-likelihood
  for (n in 1:N) {
    target += normal_lpdf(t[n] | beta[1] + beta[2]*age[n] + beta[3]*gender[n] + alpha_race[race[n]] + alpha_state[state[n]] + alpha_crime[crime[n]], sigma);
  }
  // Priors
  target +=  normal_lpdf(alpha_race | 0, sigma_race);
  target +=  normal_lpdf(alpha_state | 0, sigma_state);
  target +=  normal_lpdf(alpha_crime | 0, sigma_crime);
  target += normal_lpdf(beta | 0, 1);
  target += normal_lpdf(sigma_race | 0, 1);
  target += normal_lpdf(sigma_state | 0, 1);
  target += normal_lpdf(sigma_crime | 0, 1);
  target += normal_lpdf(sigma | 0, 1);
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] log_weight_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real mu = beta[1] + beta[2]*age[n] + beta[3]*gender[n] + alpha_race[race[n]] + alpha_state[state[n]] + alpha_crime[crime[n]];
    log_lik[n] = normal_lpdf(t[n] | mu, sigma);
    log_weight_rep[n] = normal_rng(mu, sigma);
  }
}