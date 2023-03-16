import numpy as np
from sklearn.preprocessing import StandardScaler


def gen_lin_data(d: int, synth_gauss: dict):
    '''
    This method generates data from the data generating process described above.
    ---------------------------------------------------------------------------
        Input: data dimension (int), snyth gauss (dictionary)
        Output: X, Y (np.arrays); training set size (int)
    '''
    
    n = synth_gauss['n']
    n_train = int(n / 2)
    print(f'Number of training samples: {n_train}')
    
    sigma2 = synth_gauss['sigma2']
    frac_important_features = synth_gauss['frac_important_features']
    d_relevant = int(np.floor(d * frac_important_features))
    mu = np.zeros(d)
    Sigma2 = np.ones((d, d)) * synth_gauss['corr'] + np.eye(d) * sigma2
    
    ### true parameter vector ###
    beta0 = np.zeros(d)
    beta = np.random.random(d) * 2 - 1  # Coefficient randomly chosen from [-1,1]^d
    quantile = np.quantile(np.abs(beta), 1 - frac_important_features)
    indeces = np.where(np.abs(beta) >= quantile)[0]  # Determine indeces of relevant coefficients
    
    beta0[0:d_relevant] = beta[indeces]
    beta1 = beta0 / np.linalg.norm(beta0, ord=2)  # Normalize to have unit vector 1
    
    ### error variance parameters & distribution of errors ###
    sigma2_eps = synth_gauss['sigma2_eps']
    sigma_eps = np.sqrt(sigma2_eps)
    eps = np.random.normal(0, sigma_eps, n)  # Parameterzied in terms of sigma (not sigma2)
    
    ### data generation ###
    X = np.random.multivariate_normal(mu, Sigma2, n)  # Generate & standardize data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    score = X @ beta1 + eps  # Generate labels
    prob = 1 / (1 + np.exp(-score))
    Y_disc = (prob > 0.5) * 1
    
    return X, Y_disc
