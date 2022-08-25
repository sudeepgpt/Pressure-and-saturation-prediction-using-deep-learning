import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dill

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from pytorchtools_earlystop import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP2(nn.Module):
    def __init__(self, D_in, H, D_out, activation='relu', 
                drop=False, drop_decrease=False, 
                droprates=[0.2, 0.5]):
        super(MLP2, self).__init__()
        self.model = nn.Sequential()
        if drop:
            if droprates[0] != 0.0:
                self.model.add_module("dropout0",nn.Dropout(p=droprates[0]))
        self.model.add_module("input", nn.Linear(D_in, H[0]))
        if activation == 'relu':
            self.model.add_module("relu0", nn.ReLU())
        elif activation == 'tanh':    
            self.model.add_module("tanh0", nn.Tanh())
        
        # Add hidden layers
        droprate=droprates[1]
        for i,d in enumerate(H[:-1]):
            if drop:
                if droprates[1] != 0.0:
                    self.model.add_module("dropout_hidden"+str(i+1), nn.Dropout(p=droprate))
            self.model.add_module("hidden"+str(i+1), nn.Linear(H[i], H[i+1]))
            if activation == 'relu':
                self.model.add_module("relu"+str(i+1), nn.ReLU())
            elif activation == 'tanh':    
                self.model.add_module("tanh"+str(i+1), nn.Tanh())
            if drop_decrease:
                droprate -= 0.1
                if droprate < 0:
                    droprate =0.0
        
        if drop:
            self.model.add_module("dropout_final",nn.Dropout(p=droprate))

        self.model.add_module("final",nn.Linear(H[-1], D_out))        
        
    def forward(self, x):
        # Turn to 1D
        x = self.model(x)
        return x

class MLP2Regressor:
    def __init__(self, D_in, H, D_out, activation='relu',
                drop=False, drop_decrease=False, 
                droprates=[0.2, 0.5],
                max_epoch=10, lr=0.0001, weight_decay=1e-6,
                outpath='./'):
        self.D_in = D_in
        self.D_out = D_out
        self.H = H
        self.activation = activation
        self.drop = drop
        self.drop_decrease = drop_decrease
        self.droprates=droprates     
        self.max_epoch = max_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.outpath = outpath

        self.model = MLP2(D_in=D_in, H=H, D_out=D_out, activation=activation,
                        drop=drop, drop_decrease=drop_decrease, 
                        droprates=droprates)
        self.model.to(device)
        self.criterion = nn.SmoothL1Loss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        try:
            os.remove(os.path.join(self.outpath,'run.log'))
            print('... Remove log ...')
        except:
            print('... New Run ...')

    def save_mod(self,losses, epoch, outfile):
        mod_save = {}
        mod_save['losses'] = losses[0:epoch]
        mod_save['D_in'] = self.D_in
        mod_save['D_out'] = self.D_out
        mod_save['H'] = self.H
        mod_save['lr'] = self.lr
        mod_save['drop'] = self.drop
        mod_save['drop_decrease'] = self.drop_decrease
        mod_save['droprates'] = self.droprates
        mod_save['activation'] = self.activation
        mod_save['mod_state_dict'] = self.model.state_dict()
        mod_save['mod_optim_state_dict'] = self.optimizer.state_dict()
        torch.save(mod_save, outfile, pickle_module=dill)

    def fit_accu_MSLR_EarlyStopSelect(self, X_train, y_train, X_val, y_val, batch_size, 
                                    stop_nepoch,schedule=True,stopping=True,verbose=True):
        X = Variable(X_train)
        Y = Variable(y_train)

        Xv = Variable(X_val)
        yv = Variable(y_val)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size
        )

        losses = np.empty((self.max_epoch))
        accuracy = np.empty((self.max_epoch))
        learnrates = np.empty((self.max_epoch))
        stopped = -1*np.ones((self.max_epoch))

        if schedule:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, 
                                steps_per_epoch=len(loader), epochs=self.max_epoch)
        
        early_stopping = EarlyStopping(patience=stop_nepoch, verbose=True)
        
        for epoch in range(self.max_epoch):
            r_loss = 0.0
            for batch_idx, (x, y) in enumerate(loader):
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                r_loss += float(loss.item())
                if schedule:
                    scheduler.step()
                del loss

            losses[epoch] = r_loss

            ### Validate
            with torch.no_grad():
                self.model.eval()
                val = self.model(Xv)
                accuracy[epoch] = self.criterion(val,yv)
                accuracy[epoch] = mean_squared_error(yv.cpu(), val.cpu())
                
            
            learnrates[epoch] = self.optimizer.param_groups[0]['lr']

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(accuracy[epoch], self.model)
            stopped[epoch] = early_stopping.counter

            if stopping:            
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            self.model.train()
            ### LR scheduler
            #if verbose:
            #    if epoch % (int(self.max_epoch/100)) == 0:
            #        print('Epoch {} loss: {} val_loss: {} lr: {}'.format(epoch+1, r_loss, 
            #                            accuracy[epoch],self.optimizer.param_groups[0]['lr']))
        return self, losses, accuracy, learnrates, stopped

    def predict(self, x):
        # Used to keep all test errors after each epoch
        model = self.model.eval()
        outputs = model(Variable(x))
        _, pred = torch.max(outputs.data, 1)
        model = self.model.train()
        return pred
