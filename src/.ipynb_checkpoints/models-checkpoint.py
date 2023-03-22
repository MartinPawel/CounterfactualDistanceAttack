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


def fit_model(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, params: dict, loss: str = 'log'):
    
    if params['linear']:
        # FIT LINEAR MODEL
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
                # print('training set accuracy on last ensemble model:', clf[-1].score(X_train_, Y_train_))
                # print('test set accuracy across all models:', score_test)    

            else:
                clf = LogisticRegression(penalty=params['penalty'], C=params['C'], fit_intercept=True, max_iter=2000)
                clf.fit(X_train, Y_train)

                # print('training set accuracy:', clf.score(X_train, Y_train))
                # print('test set accuracy:', clf.score(X_test, Y_test))
    else:
        # FIT RELU MODEL
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
        # print(X_train.shape)
        clf.fit(X=X_train,
                y=Y_train[:].reshape(-1))

        pred_train = ((clf(torch.from_numpy(X_train).float()) > 0.5) * 1).detach().numpy()
        pred_test = ((clf(torch.from_numpy(X_test).float()) > 0.5) *1 ).detach().numpy()

        # print("Accuracy on Train set:", accuracy_score(Y_train, pred_train))
        # print("Accuracy on Test set:", accuracy_score(Y_test, pred_test))
    return clf

    
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
        # print("Loss:", sklearn_ann.loss_)

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

    
    
class VAE_model(nn.Module):
    
    def __init__(self, 
                 D: int, 
                 activFun=nn.Softplus(), 
                 d: int = 6,
                 H1: int = 25, 
                 H2: int = 25):
        super(VAE_model, self).__init__()

        # The VAE components
        self.enc = nn.Sequential(
            nn.Linear(D, H1),
            activFun,
            nn.Linear(H1, H2),
            activFun
        )

        self.mu_enc = nn.Sequential(
            self.enc,
            nn.Linear(H2, d)
        )

        self.log_var_enc = nn.Sequential(
            self.enc,
            nn.Linear(H2, d)
        )

        self.dec = nn.Sequential(
            nn.Linear(d, H2),
            activFun,
            nn.Linear(H2, H1),
            activFun
        )

        self.mu_dec = nn.Sequential(
            self.dec,
            nn.Linear(H1, D)
        )

        self.log_var_dec = nn.Sequential(
            self.dec,
            nn.Linear(H1, D)
        )

    def encode(self, x):
        return self.mu_enc(x), self.log_var_enc(x)
    
    def encode_csearch(self, x):
        return self.mu_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)
    
    def decode_csearch(self, z):
        return self.mu_dec(z)

    @staticmethod
    def reparametrization_trick(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z_rep = self.reparametrization_trick(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z_rep)

        return mu_x, log_var_x, z_rep, mu_z, log_var_z

    def predict(self, data):
            return self.forward(data)
    
    def regenerate(self, z, grad=False):
        mu_x, log_var_x = self.decode(z)
        return mu_x

    # Computes the objective function of the VAE
    def VAE_loss(self, x, mu_x, log_var_x, mu_z, log_var_z, r=0.25):
        D = mu_x.shape[1]
        d = mu_z.shape[1]

        if log_var_x.shape[1] == 1:
            P_X_Z = + 0.5 * (D * log_var_x + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()
        else:
            P_X_Z = + 0.5 * (log_var_x.sum(dim=1, keepdim=True)
                            + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()

        if log_var_z.shape[1] == 1:
            Q_Z_X = - 0.5 * (d * log_var_z).mean()
        else:
            Q_Z_X = - 0.5 * log_var_z.sum(dim=1, keepdim=True).mean()

        if log_var_z.shape[1] == 1:
            P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + d * log_var_z.exp()).mean()
        else:
            P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + log_var_z.exp().sum(dim=1, keepdim=True)).mean()

        return P_X_Z + r * Q_Z_X + r * P_Z
    
    
    def fit(self, 
            model, 
            train_loader, 
            test_loader, 
            learning_rate=0.002, 
            epochs=50, 
            batch_size=32, 
            lambda_reg=1e-6):
    
        loaders = {'train': train_loader,
                   'test': test_loader}

        optimizer_model = torch.optim.Adam(model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=lambda_reg)

        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Train the VAE with the new prior
        ELBO_train = np.zeros((epochs, 1))
        ELBO_test = np.zeros((epochs, 1))

        for epoch in range(epochs):

            if epoch % 5 == 0:
                print('-' * 10)
                print('Epoch {}/{}'.format(epoch, epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation mode

                # Initialize the losses
                train_loss = 0
                test_loss = 0

                # Train for all the batches
                for batch_idx, (data, _, _) in enumerate(loaders[phase]):
                    data = data.view(data.shape[0], -1).float()

                    optimizer_model.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(data)

                        # The VAE loss
                        loss = model.VAE_loss(x=data, mu_x=MU_X_eval, log_var_x=LOG_VAR_X_eval,
                                              mu_z=MU_Z_eval, log_var_z=LOG_VAR_Z_eval)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_model.step()
                            train_loss += loss.detach().item() / batch_size
                        else:
                            test_loss += loss.detach().item() / batch_size

                if epoch % 10 == 0 and phase == 'train':
                    ELBO_train[epoch] = train_loss
                    print("[Epoch: {}| {}/{}] [ELBO: {:.3f}]".format(phase, epoch, epochs, ELBO_train[epoch, 0]))
                elif epoch % 10 == 0 and phase == 'test':
                    ELBO_test[epoch] = test_loss
                    print("[Epoch: {}| {}/{}] [ELBO: {:.3f}]".format(phase, epoch, epochs, ELBO_test[epoch, 0]))

        print("Training on completed")
        
        
def train_vae(state,
              X_train: np.array,
              y_train: np.array,
              X_test: np.array,
              y_test: np.array):
    
    dataset_train = Loader(X=X_train, y=y_train[:].reshape(-1))
    dataset_test = Loader(X=X_test, y=y_test[:].reshape(-1))
    trainloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=32, shuffle=True)  
    model_private = VAE_model(input_size=X_train.shape[1])
    
    # Fit (and save model since it doesn't exist, yet)
    model_private.fit(trainloader, 
                      testloader,
                      dataset_train)
    return model_private