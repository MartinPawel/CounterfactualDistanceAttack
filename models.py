from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

def _disjoint_datasets(data: np.array, labels: np.array, percentage: list, shuffle: bool = True):
    """
    :param patches: data patches
    :param label_patches: label patches
    :param percentage: list of percentages for each value, example [0.9, 0.02, 0.08] to get 90% train, 2% val and 8% test.
    :param shuffle: Shuffle dataset before split.
    :return: tuple of two lists of size = len(percentage), one with data x and other with labels y.
    """
    x_test = data
    y_test = labels
    percentage = list(percentage)       # need it to be mutable
    # assert sum(percentage) == 1., f"percentage must add to 1, but it adds to sum{percentage} = {sum(percentage)}"
    x = []
    y = []
    for i, per in enumerate(percentage[:-1]):
        x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=1-per, shuffle=shuffle)
        percentage[i+1:] = [value / (1-percentage[i]) for value in percentage[i+1:]]
        x.append(x_train)
        y.append(y_train)
    x.append(x_test)
    y.append(y_test)
    return x, y


def fit_linear_model(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, params: dict, loss: str = 'log'):
    if params['fit_sgd']:
        clf = SGDClassifier(loss=loss, penalty=params['penalty'], fit_intercept=True, max_iter=1500, tol=1e-6)
        clf.fit(X_train, Y_train)
    else:
        if params['ensemble']:
            clf = []
            if params['disjoint']:
                percentages = [1/params['n_splits']] * params['n_splits']
                X_trains, Y_trains = _disjoint_datasets(X_train, Y_train, percentage=percentages)
                for i in range(params['n_splits']):
                    X_train_, Y_train_ = X_trains[i], Y_trains[i] 
                    m = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2000)
                    m.fit(X_train_, Y_train_)
                    clf.append(m)
            else:
                for i in range(params['n_ensemble']):
                    X_train_, X_hold, Y_train_, Y_hold = train_test_split(X_train, 
                                                                          Y_train, 
                                                                          test_size=params['frac_ensemble'],
                                                                          random_state=567+i)

                    m = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2000)
                    m.fit(X_train_, Y_train_)
                    clf.append(m)
            
            score_test = 0
            if params['disjoint']:
                for i in range(params['n_splits']):
                    score_test += (1/params['n_splits']) * clf[i].score(X_test, Y_test)
            else:
                for i in range(params['n_ensemble']):
                    score_test += (1/params['n_ensemble']) * clf[i].score(X_test, Y_test)
            print('training set accuracy on last ensemble model:', clf[-1].score(X_train_, Y_train_))
            print('test set accuracy across all models:', score_test)    
            
        else:
            clf = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2000)
            clf.fit(X_train, Y_train)
    
            print('training set accuracy:', clf.score(X_train, Y_train))
            print('test set accuracy:', clf.score(X_test, Y_test))
    
    return clf


class Loader(data.Dataset):
    def __init__(self, X: np.array, y: np.array):
        
        """
        Load training dataset
        :param path: string with path to training set
        :param label: string, column name for label
        :param scale: string; either 'minmax' or 'standard'
        :return: tensor with training data
        """
        
        # Load dataset
        self.y= y
        # Save target and predictors
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # select correct row with idx
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return self.X[idx], self.y[idx], idx

    def get_number_of_features(self):
        return self.X.shape[1]
    
    def get_number_of_instances(self):
        return self.y.shape[0]


def fit_nonlinear_model(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, params: dict):
    dataset_train = Loader(X=X_train, 
                           y=Y_train[:].reshape(-1))
    trainloader = DataLoader(dataset_train, 
                             batch_size=params['batch_size'], 
                             shuffle=True)
    clf = ann(input_dim=X_train.shape[1], 
              hidden_layers=params['hidden_layers'],
              train_loader=trainloader,
              epochs=params['epochs'])
    
    width = len(params['hidden_layers'])
    depth = params['hidden_layers'][0] 
    print(X_train.shape)
    #print(X_train
    clf.fit(X=X_train,
            y=Y_train[:].reshape(-1))
    
    pred_train = ((clf(torch.from_numpy(X_train).float()) > 0.5) * 1).detach().numpy()
    pred_test = ((clf(torch.from_numpy(X_test).float()) > 0.5) *1 ).detach().numpy()
    
    print("Accuracy on Train set:", accuracy_score(Y_train, pred_train))
    print("Accuracy on Test set:", accuracy_score(Y_test, pred_test))
    return clf


class ann(nn.Module):    
    
    def __init__(self, input_dim: int, 
                 hidden_layers: list,
                 train_loader,
                 num_of_classes: int = 1, 
                 epochs=750):
        
        super().__init__()
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.optim = "sgd"
        self.n_epochs = epochs

        # input layer
        self.input1 = nn.Linear(input_dim, self.hidden_layers[0])

        # second layer if necessary
        if len(hidden_layers) == 2:
            self.input2 = nn.Linear(self.hidden_layers[1], self.hidden_layers[1])
            self.input3 = nn.Linear(hidden_layers[1], num_of_classes)
        else:
            # output layer
            self.input3 = nn.Linear(hidden_layers[0], num_of_classes)

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def fit(self, X: np.array, y: np.array):
        self._fit_sklearn(X, y)
    
    def _fit_sklearn(self, 
                     X: np.array, 
                     y: np.array):
        
        num_inputs = len(X)
        sklearn_ann = MLPClassifier(hidden_layer_sizes=self.hidden_layers, 
                                    solver='lbfgs',
                                    activation='relu', 
                                    alpha=0.0, 
                                    max_iter=750, 
                                    tol=1e-4)
        sklearn_ann.fit(X, y)
        self.input1.weight.data = torch.tensor(sklearn_ann.coefs_[0], dtype=torch.float32).t()
        self.input1.bias.data = torch.tensor(sklearn_ann.intercepts_[0], dtype=torch.float32).flatten()
        
        if len(self.hidden_layers) == 2:
            self.input2.weight.data = torch.tensor(sklearn_ann.coefs_[1], dtype=torch.float32).t()
            self.input2.bias.data = torch.tensor(sklearn_ann.intercepts_[1], dtype=torch.float32).flatten()
            
            self.input3.weight.data = torch.tensor(sklearn_ann.coefs_[2], dtype=torch.float32).t()
            self.input3.bias.data = torch.tensor(sklearn_ann.intercepts_[2], dtype=torch.float32).flatten()
        else:
            self.input3.weight.data = torch.tensor(sklearn_ann.coefs_[1], dtype=torch.float32).t()
            self.input3.bias.data = torch.tensor(sklearn_ann.intercepts_[1], dtype=torch.float32).flatten()
        print("Sklearn loss:", sklearn_ann.loss_)

    def forward(self, x):
        """
        Forward pass through the network
        Parameters
        ----------
        x: tabular data input
        Returns
        -------
        prediction
        """
        
        output = self.input1(x)
        output = self.relu(output)
        
        if len(self.hidden_layers) == 2:
            output = self.input2(output)
            output = self.relu(output)
        
        output = self.input3(output)
        output = self.sigmoid(output)
        return output.reshape(-1)
