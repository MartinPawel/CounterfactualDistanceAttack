import numpy as np
import datetime
import torch 
import torch.nn as nn

def vanilla_distances(score_train, score_test, w_train):
    '''
    This method takes the logit scores (as defined above) and the data weighs
    and computes standard ell_2 distances to logit target score.
    ---------------------------------------------------------------------------
        Input: f (np.array), w (np.array)
        Output: distances (np.arrays)
    '''
    n_train = score_train.shape[0]
    n_test = score_test.shape[0]
    
    train_deltas = - score_train.reshape(-1, 1) / np.linalg.norm(w_train, ord=2) ** 2 * np.tile(w_train, (n_train, 1))
    test_deltas = - score_test.reshape(-1, 1) / np.linalg.norm(w_train, ord=2) ** 2 * np.tile(w_train, (n_test, 1))
    
    train_dist = np.linalg.norm(train_deltas, ord=2, axis=1)
    test_dist = np.linalg.norm(test_deltas, ord=2, axis=1)
    
    return np.log(train_dist), np.log(test_dist)


def latent_distances(X_train, score_train, score_test, w_train, k=10, s=0):
    '''
    This method takes the logit scores (as defined above) and the data weighs
    and computes standard ell_2 distances to logit target score.
    ---------------------------------------------------------------------------
        Input: f (np.array), w (np.array)
        Output: distances (np.arrays)
    '''
    n_train = score_train.shape[0]
    n_test = score_test.shape[0]
    
    ### Do PCA
    cov = (X_train.T @ X_train) / (X_train.shape[0] - 1)
    eig_values, eig_vectors = np.linalg.eig(cov)
    idx = np.argsort(eig_values, axis=0)[::-1]
    sorted_eig_vectors = eig_vectors[:, idx]
    W = sorted_eig_vectors[:, :k]
    
    ### Compute latent codes & reconstructions
    # Z_train = np.dot(X_train, W)
    # Xhat_train = np.dot(Z_train, W.T)
    # Z_test = np.dot(X_test, W)
    # Xhat_test = np.dot(Z_test, W.T)
    
    ### Compute distances
    ry_train = s - score_train
    ry_test = s - score_test
    w_tilde = W.T @ w_train
    delta_z_train = (ry_train.reshape(-1, 1) / np.linalg.norm(w_tilde, ord=2) ** 2) * np.tile(w_tilde, (n_train, 1))
    train_deltas = delta_z_train @ W.T
    delta_z_test = (ry_test.reshape(-1, 1) / np.linalg.norm(w_tilde, ord=2) ** 2) * np.tile(w_tilde, (n_test, 1))
    test_deltas = delta_z_test @ W.T
    train_dist = np.linalg.norm(train_deltas, ord=2, axis=1)
    test_dist = np.linalg.norm(test_deltas, ord=2, axis=1)
    
    return np.log(train_dist), np.log(test_dist)


class SCFE:
    def __init__(self, 
                 classifier, 
                 target_threshold: float = 0.5, 
                 _lambda: float = 10.0,
                 lr: float = 0.05,
                 max_iter: int = 500, 
                 t_max_min: float = 0.15,
                 step: float = 0.25,
                 norm: int = 1, 
                 optimizer: str = 'adam'):
        
        super().__init__()
        self.model_classification = classifier
        self.lr = lr
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.t_max_min = t_max_min
        self.norm = norm
        self.sigmoid = nn.Sigmoid()
        self.target_thres = target_threshold
        self._lambda = _lambda
        self.step = step
    
    def generate_counterfactuals(self, 
                                 query_instance: torch.tensor, 
                                 target_class: int = 1) -> torch.tensor:
        """
            query instance: the point to be explained
            target_class: Direction of the desired change. If target_class = 1, we aim to improve the score,
            if target_class = 0, we aim to decrese it (in classification and regression problems).
            _lambda: Lambda parameter (distance regularization) parameter of the problem
        """
        
        if target_class == 1:
            target_prediction = torch.tensor(1).float().reshape(-1)
        else:
            target_prediction = torch.tensor(0).float().reshape(-1)
        
        output = self._call_model(query_instance.reshape(1, -1))
        check_output = output.clone().detach()
        
        cf = query_instance.clone().requires_grad_(True)
        
        if self.optimizer == 'adam':
            optim = torch.optim.Adam([cf], self.lr)
        else:
            optim = torch.optim.RMSprop([cf], self.lr)
        
        # Timer
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=self.t_max_min)
        
        counterfactuals = []
        while not self._check_cf_valid(output, target_class):
            
            it = 0
            distances = []
            all_loss = []
            
            while not self._check_cf_valid(output, target_class) and it < self.max_iter:
                cf.requires_grad = True
                optim.zero_grad()
                total_loss, loss_distance = self.compute_loss(self._lambda,
                                                              cf,
                                                              query_instance,
                                                              target_prediction)
                total_loss.backward()
                optim.step()                
                output = self._call_model(cf)
                
                if self._check_cf_valid(output, target_class):
                    counterfactuals.append(cf.clone().detach())
                    dist = torch.norm(cf - query_instance, self.norm).clone().detach()
                    distances.append(dist)
                    all_loss.append(total_loss.clone().detach())
                
                it = it + 1
            
            output = self._call_model(cf).reshape(1, -1).detach()
            if datetime.datetime.now() - t0 > t_max:
                break

            if self.step == 0.0:  # Don't search over lambdas
                break
            else:
                self._lambda -= self.step

        if not len(counterfactuals):
            # print('No CE found')
            cf.detach_()
            return torch.tensor(np.nan)
        
        # Choose the nearest counterfactual
        counterfactuals = torch.stack(counterfactuals)
        
        distances = torch.stack(distances)
        distances = distances.detach()
        index = torch.argmin(distances)
        counterfactuals = counterfactuals.detach()

        ce_star = counterfactuals[index]
        distance_star = distances[index]
        return distance_star.numpy()
    
    def compute_loss(self, 
                     _lambda: float, 
                     cf_candidate: torch.tensor, 
                     original_instance: torch.tensor,
                     target: torch.tensor) -> torch.tensor:
        
        output = self._call_model(cf_candidate)
        bce_loss = nn.BCELoss()
        loss_classification = bce_loss(output.reshape(-1), target)
        loss_distance = torch.norm((cf_candidate - original_instance), self.norm)
        total_loss = loss_classification + _lambda * loss_distance
        return total_loss, loss_distance

    def _call_model(self, cf_candidate):
        output = self.model_classification(cf_candidate)[0]
        return output

    def _check_cf_valid(self, output, target_class):
        """ Check if the output constitutes a sufficient CF-example.
            target_class = 1 in general means that we aim to improve the score,
            whereas for target_class = 0 we aim to decrease it.
        """
        if target_class == 1:
            check = output > self.target_thres
            return check
        else:
            check = output < self.target_thres
            return check

def nonvanilla_distances(X_train,
                         X_test,
                         model,
                         parameters: dict,
                         recourse_type: str = 'scfe'):
    preds_train1 = model(torch.from_numpy(X_train).float()).detach().numpy()
    preds_test1 = model(torch.from_numpy(X_test).float()).detach().numpy()
    # choose recourse model
    if recourse_type == 'scfe':
        recourse_model = SCFE(classifier=model,
                              lr=parameters['lr_scfe'],
                              _lambda=0.0,
                              step=0.00,
                              max_iter=parameters['max_iter'],
                              norm=1,
                              target_threshold=0.5)
    else:
        raise ValueError('This version of the code currently only supports SCFE')
    # generate distances
    train_dist = gen_distances(X_train, preds_train1, recourse_model, recourse_type)
    test_dist = gen_distances(X_test, preds_test1, recourse_model, recourse_type)  
    return np.log(train_dist), np.log(test_dist)


def gen_distances(X: np.array, preds: np.array, recourse_model, recourse_type):
    distances = np.zeros_like(X[:, 0])
    ## TODO: parallelize this by batchting
    for j in range(X.shape[0]):
        pred_class = torch.tensor((preds[j] > 0.5) * 1)
        q_j = torch.tensor(X[j,:]).reshape(1,-1).float()
        distance = recourse_model.generate_counterfactuals(query_instance=q_j,
                                                           target_class=1-pred_class)
        distances[j] = distance
        if (j % 100) == 0:
            print(f"Finding counterfactual for {recourse_type} at iteration: {j}")
    return distances