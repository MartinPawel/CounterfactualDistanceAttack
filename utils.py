# distribution stuff
import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import t

# model stuff
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import QuantileRegressor
import torch.utils.data as data
from torch.utils.data import DataLoader
from models import fit_nonlinear_model

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
    # COMPUTE PREDICTIONS#
    #####################
    
    ### predictions
    scores['preds_train'].append(
        np.log(y_pred_train1 + epsd) - np.log(y_pred_train0 + epsd))
    scores['preds_test'].append(
        np.log(y_pred_test1 + epsd) - np.log(y_pred_test0 + epsd))
    
    ### LRT predictions
    shadow_preds_train, shadow_preds_test = compute_shadowpredictions(X_train, X_test,
                                                                      X_test, Y_test,
                                                                      params)

    
    # GLOBAL VAR
    # use absolute values of logits as scores
    vars_global_preds = np.var(np.r_[shadow_preds_train, shadow_preds_test], axis=1)
    if params['weighting'] == 'equal':
        var_global_preds = 1
    else:
        var_global_preds = np.mean(vars_global_preds)
    
    preds_lrt_score_train = compute_lambda_predictions(
        np.abs(np.log(y_pred_train1 + epsd) - np.log(y_pred_train0 + epsd)),
        shadow_preds_train,
        var_global_preds,
        global_variance=True)
    
    preds_lrt_score_test = compute_lambda_predictions(np.abs(np.log(y_pred_test1 + epsd) - np.log(y_pred_test0 + epsd)),
                                                      shadow_preds_test,
                                                      var_global_preds,
                                                      global_variance=True)
    
    scores['preds_lrt_train_global'].append(preds_lrt_score_train)
    scores['preds_lrt_test_global'].append(preds_lrt_score_test)
    
    # LOCAL VAR
    preds_lrt_score_train = compute_lambda_predictions(
        np.abs(np.log(y_pred_train1 + epsd) - np.log(y_pred_train0 + epsd)),
        shadow_preds_train,
        var_global_preds,
        global_variance=False)
    
    preds_lrt_score_test = compute_lambda_predictions(np.abs(np.log(y_pred_test1 + epsd) - np.log(y_pred_test0 + epsd)),
                                                      shadow_preds_test,
                                                      var_global_preds,
                                                      global_variance=False)
    
    scores['preds_lrt_train_local'].append(preds_lrt_score_train)
    scores['preds_lrt_test_local'].append(preds_lrt_score_test)
    
    #####################
    # COMPUTE DISTANCES #
    #####################
    ### vanilla distances
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

        train_dist, test_dist = vanilla_distances(f_train, f_test, w_train)
        train_dist_latent, test_dist_latent = latent_distances(X_train, f_train, f_test, w_train, k=5, s=0)
    
    else:
        '''
        here we use different recourse methods to approximate distance to boundary since the unerlying model is nonlinear
        '''
        clf = fit_nonlinear_model(X_train, X_test, Y_train, Y_test, params)
        train_dist, test_dist = nonvanilla_distances(X_train, X_test, clf, params, 'scfe')
        
    ### LRT distances
    shadow_dists_train, shadow_dists_test = compute_shadowdistances(X_train, X_test,
                                                                    X_test, Y_test,
                                                                    params)


    # GLOBAL VAR
    vars_global_dists = np.var(np.r_[shadow_preds_train, shadow_preds_test], axis=1)
    if params['weighting'] == 'equal':
        var_global_dists = 1
    else:
        var_global_dists = np.mean(vars_global_preds)
    
    dists_lrt_score_train = compute_lambda_predictions(train_dist,
                                                       shadow_dists_train,
                                                       var_global_dists,
                                                       global_variance=True)
    
    dists_lrt_score_test = compute_lambda_predictions(test_dist,
                                                      shadow_dists_test,
                                                      var_global_dists,
                                                      global_variance=True)
    
    # standard
    scores['dists_train'].append(train_dist)
    scores['dists_test'].append(test_dist)
    scores['dists_latent_train'].append(train_dist_latent)
    scores['dists_latent_test'].append(test_dist_latent)
    # lrt
    scores['dists_lrt_train_global'].append(dists_lrt_score_train)
    scores['dists_lrt_test_global'].append(dists_lrt_score_test)
    
    # LOCAL VAR
    dists_lrt_score_train = compute_lambda_predictions(train_dist,
                                                       shadow_dists_train,
                                                       var_global_dists,
                                                       global_variance=False)
    
    dists_lrt_score_test = compute_lambda_predictions(test_dist,
                                                      shadow_dists_test,
                                                      var_global_dists,
                                                      global_variance=False)
    
    scores['dists_lrt_train_local'].append(dists_lrt_score_train)
    scores['dists_lrt_test_local'].append(dists_lrt_score_test)
    
    '''
    ### quantile distances
    # STEP 1: GET DISTANCES
    if params['ensemble']:
        y_pred_prime1 = y_pred_prime0 = 0
        for i in range(params['n_ensemble']):
            y_pred_prime1 += (1/params['n_ensemble']) * clf[i].predict_proba(X_prime)[:, 1]
            y_pred_prime0 += (1/params['n_ensemble']) * clf[i].predict_proba(X_prime)[:, 0]
    
    else:
        y_pred_prime1 = clf.predict_proba(X_prime)[:, 1]
        y_pred_prime0 = clf.predict_proba(X_prime)[:, 0]

    f_test_prime = np.log(y_pred_prime1 + epsd) - np.log(y_pred_prime0 + epsd)
    prime_dist, _ = vanilla_distances(f_test_prime, f_test_prime, w_train)

    # STEP 2: GET QUANTILES
    qr = QuantileRegressor(quantile=params['quantile'], alpha=0, solver="highs")
    clf_qr = qr.fit(X_prime, prime_dist)

    nf_quantile_train = clf_qr.predict(X_train)
    nf_quantile_test = clf_qr.predict(X_test)

    measure_train = train_dist - nf_quantile_train
    measure_test = test_dist - nf_quantile_test

    scores['dists_train_quantile'].append(measure_train)
    scores['dists_test_quantile'].append(measure_test)
    '''
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
            model = fit_nonlinear_model(X_prime_shadow, X_prime_shadow, Y_prime_shadow, Y_prime_shadow, params)
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
            model = fit_nonlinear_model(X_prime_shadow, X_test, Y_prime_shadow, Y_test, params)
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


def compute_shadowdistances(X_train, X_test, X_shadow, Y_shadow, params, epsd=1e-5):
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
            model = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2500).fit(X_prime_shadow, Y_prime_shadow)
            f_train = np.log(model.predict_proba(X_train)[:, 1] + epsd) - np.log(model.predict_proba(X_train)[:, 0] + epsd)
            w_train = model.coef_[0]
            f_test = np.log(model.predict_proba(X_test)[:, 1] + epsd) - np.log(model.predict_proba(X_test)[:, 0] + epsd)
            train_dist, test_dist = vanilla_distances(f_train, f_test, w_train)
        else:
            model = fit_nonlinear_model(X_prime_shadow, X_test, Y_prime_shadow, Y_test, params)
            train_dist, test_dist = nonvanilla_distances(X_train, X_test, model, params, 'scfe')
        distances_train[:, i] = train_dist
        distances_test[:, i] = test_dist
    return distances_train, distances_test
