# distribution stuff
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.stats import t

# model stuff
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import QuantileRegressor
import torch.utils.data as data
from torch.utils.data import DataLoader
from models import fit_model

# local imports
from counterfactuals import vanilla_distances
from counterfactuals import nonvanilla_distances
from counterfactuals import latent_distances


def pipeline(clf, X_train, X_test, X_prime, Y_train, Y_test, Y_prime, scores, params):
    
    """
    TODO: MAKE THIS MORE GENERAL & CONVERT all TO PYTORCH:
    NOTE:
    if: model is linear: clf is sklearn instance
    else: pytorch
    """
    
    ####################
    ## COMPUTE LOSSES ##
    ####################

    epsd = params['epsd']
    
    if params['linear']:
        if params['ensemble']:
            y_pred_train1 = y_pred_train0 = y_pred_test1 = y_pred_test0 = 0
            for i in range(params['n_ensemble']):
                y_pred_train1 += (1/params['n_ensemble']) * clf[i].predict_proba(X_train)[:, 1]
                y_pred_train0 += (1/params['n_ensemble']) * clf[i].predict_proba(X_train)[:, 0]
                y_pred_test1 += (1/params['n_ensemble']) * clf[i].predict_proba(X_test)[:, 1]
                y_pred_test0 += (1/params['n_ensemble']) * clf[i].predict_proba(X_test)[:, 0]
        else:
            y_pred_train1 = clf.predict_proba(X_train)[:, 1]
            y_pred_train0 = clf.predict_proba(X_train)[:, 0]
            y_pred_test1 = clf.predict_proba(X_test)[:, 1]
            y_pred_test0 = clf.predict_proba(X_test)[:, 0]
    else:
            y_pred_train1 = clf(torch.from_numpy(X_train).float()).detach().numpy()
            y_pred_train0 = 1-clf(torch.from_numpy(X_train).float()).detach().numpy()
            y_pred_test1 = clf(torch.from_numpy(X_test).float()).detach().numpy()
            y_pred_test0 = 1-clf(torch.from_numpy(X_test).float()).detach().numpy()
    ### vanilla losses
    train_loss = Y_train * np.log(y_pred_train1 + epsd) + (1 - Y_train) * np.log(y_pred_train0 + epsd)
    test_loss = Y_test * np.log(y_pred_test1 + epsd) + (1 - Y_test) * np.log(y_pred_test0 + epsd)
    
    scores['losses_train'].append(train_loss)
    scores['losses_test'].append(test_loss)

    ### stable losses
    stable_train_loss = np.zeros(y_pred_train1.shape[0])
    stable_test_loss = np.zeros(y_pred_test1.shape[0])
    for i in range(y_pred_train1.shape[0]):
        stable_train_loss[i] = _get_stable_logit_loss(Y_train[i], y_pred_train1[i])
    for i in range(y_pred_test1.shape[0]):
        stable_test_loss[i] = _get_stable_logit_loss(Y_test[i], y_pred_test1[i])
    
    scores['stable_losses_train'].append(stable_train_loss)
    scores['stable_losses_test'].append(stable_test_loss)
    
    ### LRT losses
    shadow_losses_train, shadow_losses_test = compute_shadowlosses(X_train, Y_train,
                                                                   X_test, Y_test,
                                                                   params)

        
    vars_global_losses = np.var(np.r_[shadow_losses_train, shadow_losses_test], axis=1)
    if params['weighting'] == 'equal':
        var_global_loss = 1
    else:
        var_global_loss = np.mean(vars_global_losses)
    loss_lrt_train = compute_lambda_loss(stable_train_loss,
                                         shadow_losses_train,
                                         var_global_loss,
                                         global_variance=True)
    
    loss_lrt_test = compute_lambda_loss(stable_test_loss,
                                        shadow_losses_test,
                                        var_global_loss,
                                        global_variance=True)
    
    scores['losses_lrt_train_global'].append(loss_lrt_train)
    scores['losses_lrt_test_global'].append(loss_lrt_test)
    
    # LOCAL VAR
    loss_lrt_train = compute_lambda_loss(stable_train_loss,
                                         shadow_losses_train,
                                         var_global_loss,
                                         global_variance=False)
    
    loss_lrt_test = compute_lambda_loss(stable_test_loss,
                                        shadow_losses_test,
                                        var_global_loss,
                                        global_variance=False)
    
    scores['losses_lrt_train_local'].append(loss_lrt_train)
    scores['losses_lrt_test_local'].append(loss_lrt_test)
    
    #####################
    # COMPUTE DISTANCES #
    #####################
    assert params['methods'] is not None, "Methods list cannot be empty: choose at least one of the following: {scfe, gs, cchvae}" 
    ### compute vanilla distances
    if params['linear']:
        '''
        # here we use closed-form solutions for \ell_2 distance to boundary since the unerlying model is linear #
        '''
        f_train = np.log(y_pred_train1 + epsd) - np.log(y_pred_train0 + epsd)
        f_test = np.log(y_pred_test1 + epsd) - np.log(y_pred_test0 + epsd)

        if params['ensemble']:
            w_train = np.zeros(X_train.shape[1])
            for i in range(params['n_ensemble']):
                w_train += clf[i].coef_[0]
        else:
            w_train = clf.coef_[0]
        if 'scfe' in params['methods']:
            # scfe
            train_dist, test_dist = vanilla_distances(f_train, f_test, w_train)
            scores['dists_train_scfe'].append(train_dist)
            scores['dists_test_scfe'].append(test_dist)
        if 'cchvae' in params['methods']:
            # latent (CCHVAE)
            train_dist_latent, test_dist_latent = latent_distances(X_train, f_train, f_test, w_train, k=5, s=0)
            scores['dists_train_cchvae'].append(train_dist_latent)
            scores['dists_test_cchvae'].append(test_dist_latent)
        if 'gs' in params['methods']:
            # gs
            train_dist_gs, test_dist_gs = nonvanilla_distances(X_train, X_test, clf, params, 'gs')
            scores['dists_train_gs'].append(train_dist_gs)
            scores['dists_test_gs'].append(test_dist_gs)
    else:
        '''
        here we use different recourse methods to approximate distance to boundary since the unerlying model is nonlinear
        '''        
        # Standard distances
        distances_train = {}
        distances_test = {}
        
        for method in params['methods']:
            train_dist, test_dist = nonvanilla_distances(X_train, X_test, clf, params, method)
            distances_train[method] = train_dist 
            distances_test[method] = test_dist 
            scores['dists_train_' + method].append(distances_train[method])
            scores['dists_test_' + method].append(distances_test[method])
        
    ### compute LRT distances
    lrt_scores_train_global = {}
    lrt_scores_test_global = {}
    lrt_scores_train_local = {}
    lrt_scores_test_local = {}
    for method in params['methods']:
        print(method)        
        shadow_dists_train, shadow_dists_test = compute_shadowdistances(X_train, X_test,
                                                                        X_test, Y_test, Y_test,
                                                                        params, method)
        # GLOBAL VAR
        vars_global_dists = np.var(np.r_[shadow_dists_train, shadow_dists_test], axis=1)
        if params['weighting'] == 'equal':
            var_global_dists = 1
        else:
            var_global_dists = np.mean(vars_global_dists)

        dists_lrt_score_train = compute_lambda_predictions(train_dist,
                                                           shadow_dists_train,
                                                           var_global_dists,
                                                           global_variance=True)

        dists_lrt_score_test = compute_lambda_predictions(test_dist,
                                                          shadow_dists_test,
                                                          var_global_dists,
                                                          global_variance=True)
        lrt_scores_train_global[method] = dists_lrt_score_train
        lrt_scores_test_local[method] = dists_lrt_score_test
        
        # lrt global
        scores['dists_lrt_train_global_' + method].append(dists_lrt_score_train)
        scores['dists_lrt_test_global_' + method].append(dists_lrt_score_test)

        # LOCAL VAR
        dists_lrt_score_train = compute_lambda_predictions(train_dist,
                                                           shadow_dists_train,
                                                           var_global_dists,
                                                           global_variance=False)

        dists_lrt_score_test = compute_lambda_predictions(test_dist,
                                                          shadow_dists_test,
                                                          var_global_dists,
                                                          global_variance=False)
        lrt_scores_train_local[method] = dists_lrt_score_train
        lrt_scores_test_local[method] = dists_lrt_score_test

        # lrt local
        scores['dists_lrt_train_local_' + method].append(dists_lrt_score_train)
        scores['dists_lrt_test_local_' + method].append(dists_lrt_score_test)

    return scores


def _get_stable_logit_loss(label,
                           prediction,
                           eps=1e-5):
    if label == 1:
        stable_logit_loss = np.log(prediction + eps) - np.log((1-prediction) + eps)
    else:
        stable_logit_loss = np.log((1 - prediction) + eps) - np.log(prediction + eps)
    return stable_logit_loss


def compute_lambda_loss(stable_losses: np.array,
                        shadow_losses: np.array,
                        global_var: float,
                        global_variance=True,
                        eps=1e-5):
    '''
    Note that: logit_loss is the converted model confidence.
    The larger the target model’s confidence (i.e., stable loss) is compared to \mu_out
    (i.e., mean_shadow_losses), the higher the likelihood that the query sample is a member:

    Likelihood that stable loss is part of train: 1 - P(Mean Shadow models (RV) \leq stable loss):

    Note that we compute the following here:
    P(Mean Shadow models (RV) \leq stable loss) = \Phi( (Mean Shadow model - stable loss) / std(Mean Shadow models) )

    Thus, later on we have to adjust -> pos_label=0 in 'compute curve method'.
    
    --------
    # TODO:
    SAME LOGIC IS TRUE FOR THE METHODS BELOW: TWO METHODS BELOW ARE NOW JUST COPIES
    -> GET RID OF THEM
    -------
    
    -------------------------------
    >>> After publication: NOTE <<<
    -------------------------------
    In this version of the code we have adjusted the test slightly: convince yourself that the test from Carlini et al (2021) is a t-test when 
    the shadow models are are trained on disjoint subsets (otherwise it may approx be a t test). In this case, the mean of the shadow losses should 
    converge to a Normal (by CLT); we know that the variance of the shadow model losses, however, is also a RV (and not constant). This RV follows a 
    Chi-sqaured distribution. We know that mean and variance are independent (for normal distributions), and thus the test statistics should follow a
    t-distribution under the  null hyothesis of 'no train membership'. Finally, observe that if the number of shadow models is ''large enough'' 
    (say large than 40 or 50), then the t-distribution can be well approximated by a normal distribution, and then the implementation here should coincide 
    with what one would get by using a normal distribution below. Feel free to change "t.cdf" to "norm.cdf", but make sure to remove the 
    degrees of freedom parameter, as the normal distribution is not parameterized by df.
    '''
    
    mean_all_shadow_losses = np.mean(shadow_losses, axis=1)
    var_all_shadow_losses = np.var(shadow_losses, axis=1, ddof=1)
    if global_variance:
        var = global_var
    else:
        var = var_all_shadow_losses + eps
    Z = ((mean_all_shadow_losses - stable_losses) / np.sqrt(var)) #* np.sqrt(shadow_losses.shape[1])
    cap_lambda = t.cdf(x=Z, loc=0, scale=1, df=shadow_losses.shape[1]-1)
    
    return cap_lambda #_check


def compute_lambda_predictions(predictions: np.array,
                               shadow_predictions: np.array,
                               global_var: float,
                               global_variance=True,
                               eps=1e-5):

    mean_all_shadow_predictions = np.mean(shadow_predictions, axis=1)
    var_all_shadow_predictions = np.var(shadow_predictions, axis=1, ddof=1)

    '''
    Note that: logit_loss is the converted model confidence.
    The larger the target model’s confidence (i.e., logit loss) is compared to µout
    (i.e., mean_shadow_losses), the higher the likelihood that the query sample is a member.
    '''
    if global_variance:
        var = global_var
    else:
        var = var_all_shadow_predictions + eps
    
    Z = ((mean_all_shadow_predictions - predictions) / np.sqrt(var)) #* np.sqrt(shadow_predictions.shape[1])
    cap_lambda = t.cdf(x=Z, loc=0, scale=1, df=shadow_predictions.shape[1]-1)
    #cap_lambda = t.cdf(x=mean_all_shadow_predictions, loc=predictions, scale=np.sqrt(var), df=shadow_predictions.shape[1]-1)

    return cap_lambda


def compute_lambda_distances(distances: np.array,
                             shadow_distances: np.array,
                             global_var: float,
                             global_variance=True,
                             eps=1e-5):

    mean_all_shadow_distances = np.mean(shadow_distances, axis=1)
    var_all_shadow_distances = np.var(shadow_distances, axis=1, ddof=1)

    '''
    Note that: logit_loss is the converted model confidence.
    The larger the target model’s confidence (i.e., logit loss) is compared to \mu_out
    (i.e., mean_shadow_losses), the higher the likelihood that the query sample is a member.
    '''
    if global_variance:
        var = global_var
    else:
        var = var_all_shadow_distances + eps
    
    Z = ((mean_all_shadow_distances - distances) / np.sqrt(var)) #* np.sqrt(shadow_distances.shape[1])
    cap_lambda = t.cdf(x=Z, loc=0, scale=1, df=shadow_distances.shape[1]-1)
    # cap_lambda = t.cdf(x=mean_all_shadow_distances, loc=distances, scale=np.sqrt(var), df=hadow_distances.shape[1]-1)

    return cap_lambda


def compute_shadowpredictions(X_train, X_test, X_shadow, Y_shadow, params, eps=1e-5):
    '''
    This method computes log probabilities for training and test points w.r.t. the shadow models.
    '''
    # Preallocate
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    predictions_train = np.zeros((n_train, params['n_shadow_models']))
    predictions_test = np.zeros((n_test, params['n_shadow_models']))
    # Compute shadow predictions
    for i in range(params['n_shadow_models']):
        ind_X_prime = np.random.choice(a=np.shape(X_shadow)[0], size=int(np.shape(X_shadow)[0] * params['frac']), replace=False)
        X_prime_shadow = X_shadow[ind_X_prime]
        Y_prime_shadow = Y_shadow[ind_X_prime]
        if params['linear']:
            model = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2500).fit(X_prime_shadow, Y_prime_shadow)
            y_pred_train1 = model.predict_proba(X_train)[:,1]
            y_pred_test1 = model.predict_proba(X_test)[:,1]
            # Compute absolute value of logit scores
        else:
            model = fit_model(X_prime_shadow, X_prime_shadow, Y_prime_shadow, Y_prime_shadow, params)
            y_pred_train1 = model(torch.from_numpy(X_train).float()).detach().numpy()
            y_pred_test1 = 1-model(torch.from_numpy(X_test).float()).detach().numpy()
        predictions_train[:,i] = np.abs(np.log(y_pred_train1 + eps) - np.log(1-y_pred_train1 + eps))
        predictions_test[:,i] = np.abs(np.log(y_pred_test1 + eps) - np.log(1-y_pred_test1 + eps))
    return predictions_train, predictions_test


def compute_shadowlosses(X_train, Y_train, X_test, Y_test, params):
    '''
    This method computes stable losses for training and test points w.r.t. the shadow models.
    '''
    # preallocate
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    losses_train = np.zeros((n_train, params['n_shadow_models']))
    losses_test = np.zeros((n_test, params['n_shadow_models']))
    # compute shadow losses
    for i in range(params['n_shadow_models']):
        ind_X_prime = np.random.choice(a=np.shape(X_test)[0], size=int(np.shape(X_test)[0] * params['frac']), replace=False)
        X_prime_shadow = X_test[ind_X_prime]
        Y_prime_shadow = Y_test[ind_X_prime]
        if params['linear']:
            model = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2500).fit(X_prime_shadow, Y_prime_shadow)
            y_pred_train1 = model.predict_proba(X_train)[:,1]
            y_pred_test1 = model.predict_proba(X_test)[:,1]
        else:
            model = fit_model(X_prime_shadow, X_test, Y_prime_shadow, Y_test, params)
            y_pred_train1 = model(torch.from_numpy(X_train).float()).detach().numpy()
            y_pred_test1 = 1-model(torch.from_numpy(X_test).float()).detach().numpy()
        # preallocate
        stable_train_loss = np.zeros(n_train)
        stable_test_loss = np.zeros(n_test)
        for j in range(n_train):
            stable_train_loss[j] = _get_stable_logit_loss(Y_train[j], y_pred_train1[j])
        for j in range(n_test):
            stable_test_loss[j] = _get_stable_logit_loss(Y_test[j], y_pred_test1[j])
        losses_train[:,i] = stable_train_loss
        losses_test[:,i] = stable_test_loss
    return losses_train, losses_test


def compute_shadowdistances(X_train, X_test, X_shadow, Y_test, Y_shadow, params, method='scfe', epsd=1e-5):
    '''
    This method computes ell_2 distances to the decision boundary
    for training and test points w.r.t. the shadow models.
    '''
    # Preallocate
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    distances_train = np.zeros((n_train, params['n_shadow_models']))
    distances_test = np.zeros((n_test, params['n_shadow_models']))
    # Compute distances: here we use close-form solution since the underlying model is linear
    for i in range(params['n_shadow_models']):
        ind_X_prime = np.random.choice(a=np.shape(X_shadow)[0], size=int(np.shape(X_shadow)[0] * params['frac']), replace=False)
        X_prime_shadow = X_shadow[ind_X_prime]
        Y_prime_shadow = Y_shadow[ind_X_prime]
        if params['linear']:
            model = fit_model(X_prime_shadow, X_test, Y_prime_shadow, Y_test, params)
            f_train = np.log(model.predict_proba(X_train)[:, 1] + epsd) - np.log(model.predict_proba(X_train)[:, 0] + epsd)
            w_train = model.coef_[0]
            f_test = np.log(model.predict_proba(X_test)[:, 1] + epsd) - np.log(model.predict_proba(X_test)[:, 0] + epsd)
            train_dist, test_dist = vanilla_distances(f_train, f_test, w_train)
        else:
            model = fit_model(X_prime_shadow, X_test, Y_prime_shadow, Y_test, params)
            train_dist, test_dist = nonvanilla_distances(X_train, X_test, model, params, method)
        distances_train[:, i] = train_dist
        distances_test[:, i] = test_dist
    return distances_train, distances_test


def remove_colls(df, corr_threshold: float = 0.8):
    # Drop multicollinear features
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than corr_threshold
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    # Drop features 
    df = df.drop(to_drop, axis=1).astype(float)
    return df


def load_data(dataname):

    if dataname == 'communities':
        df = pd.read_csv(f'../data/{dataname}.csv', header=None)
        df = df.drop_duplicates()
        df = df.replace('?', np.nan)
        df = df.drop(columns=[0, 1, 2, 3, 4, 101, 102, 103, 104, 105, 106, 107, 
                              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 
                              118, 119, 120, 121, 122, 123, 124, 126])
        df = df.dropna().astype(float)
        Y = df[127].values 
        Y = (Y > np.median(Y)) * 1
        df = df.drop(columns=[127])
        df = remove_colls(df, corr_threshold=0.90)

    elif dataname == 'housing':
        df = pd.read_csv(f'../data/{dataname}.csv')
        df = df.drop_duplicates()
        df = df.replace('?', np.nan)
        df = df.dropna()
        Y = df['median_house_value'].values 
        Y = (Y > np.median(Y))
        df = df.drop(columns=['median_house_value'])
        df['ocean_proximity'] = df['ocean_proximity'].replace(['NEAR BAY', 'ISLAND', 'NEAR OCEAN', '<1H OCEAN', 'INLAND'], 
                                                              [0, 0, 0, 1, 2])
        df = df.astype(float)

    elif dataname == "heloc":
        df = pd.read_csv(f'../data/{dataname}.csv')
        df = df.drop_duplicates()
        df = df.replace('?', np.nan)
        df = df.dropna()
        df = df.drop(columns=['RiskPerformance'])
        df = df.dropna()
        Y = df['ExternalRiskEstimate'].values 
        Y = (Y >= np.median(Y)) * 1
        df = df.drop(columns=['ExternalRiskEstimate'])
        df = df.astype(float)

    elif dataname == 'default':
        df = pd.read_csv(f'../data/{dataname}.csv', header=None)
        df = df.drop_duplicates()
        df = df.replace('?', np.nan)
        df = df.dropna()
        ind_nondefault = np.where(df.values[:,-1] == 0)[0]
        n_default = ind_nondefault.shape[0]
        ind_default = np.where(df.values[:,-1] > 0)[0]
        ind_nondefault_subset = np.random.choice(ind_nondefault, ind_default.shape[0], replace=False)
        inds = np.r_[ind_default, ind_nondefault_subset]
        df = df.iloc[inds]
        Y = (df.values[:,-1] >= np.median(df.values[:,-1])) * 1
        df = df.drop(columns=df.columns[-1])
        df = df.astype(float)
        df = remove_colls(df, corr_threshold=0.95)

    elif dataname == 'mnist':
        df_train = pd.read_csv(f'../data/{dataname}_train.csv')
        df_test = pd.read_csv(f'../data/{dataname}_test.csv')
        df = pd.concat([ df_train,  df_test], ignore_index=True)
        y0_ind = np.where(df['label']==3)[0]
        y1_ind = np.where(df['label']==8)[0]
        df['label'] = df['label'].map({3:0, 8:1})
        inds = np.r_[y0_ind , y1_ind]
        df = df.iloc[inds]
        Y = (df['label'].values) * 1
        df = df.drop(columns=['label'])
        df = remove_colls(df, corr_threshold=0.95)

    elif dataname == 'churn':
        df = pd.read_csv(f'../data/{dataname}.csv', header=None)
        df = df.drop_duplicates()
        df = df.replace('?', np.nan)
        df[230] = df[230].map({-1:0, 1:1})
        ind_nondefault = np.where(df.values[:,-1] == 0)[0]
        n_default = ind_nondefault.shape[0]
        ind_default = np.where(df.values[:,-1] > 0)[0]
        ind_nondefault_subset = np.random.choice(ind_nondefault, ind_default.shape[0], replace=False)
        inds = np.r_[ind_default, ind_nondefault_subset]
        df = df.iloc[inds]
        Y = df[230].values
        df = df.drop(columns=[230])
        df = df.iloc[:,0:190]         # keep numerical features only
        df = df.fillna(df.median())   # fill missing with median
        cols = df.columns[df.isna().any()].tolist()
        df = df.drop(columns=cols)
        df = df.dropna()
        
    return df, Y


def get_input_subset(model, 
                     inputs: torch.tensor, 
                     labels: torch.tensor,
                     subset_size: int = 500,
                     decision_threshold: float = 0.5) -> torch.tensor:
    
    """
    Get negatively classified inputs & return their predictions
    """
    pred = model(inputs).detach().numpy().reshape(-1)
    yhat = (model(inputs) > decision_threshold) * 1
    check = (model(inputs) < decision_threshold).detach().numpy()
    selected_indices = np.where(check)[0]
    input_subset = inputs[selected_indices]
    predicted_label_subset = yhat[selected_indices]
    label_subset = labels[selected_indices]
    
    return input_subset[0:subset_size, :], predicted_label_subset[0:subset_size], label_subset[0:subset_size]