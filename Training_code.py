import argparse
import time
import os
import numpy as np
#import model.data_loader as dl
import net as mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

import utils as util


def main(args):
    #### ARGS
    model           = args.model        
    rpt_test_mod    = args.rpt_test_mod 
    pressure_mod    = args.pressure_mod
    idx_P_loss      = args.idx_P_loss 
    w_P             = args.w_P 
    n_train         = args.n_train      
    n_val           = args.n_val        
    n_epochs        = args.n_epochs     
    learning_rate   = args.learning_rate
    learn_cyclic    = args.learn_cyclic
    activation      = args.activation         
    seed            = args.seed
    batch_size      = args.batch_size
    hdf_batch       = args.hdf_batch
    num_workers     = args.num_workers       
    early_stop      = args.early_stop
    n_epoch_stop    = args.n_epoch_stop

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        train_loader.num_workers = 1
        test_loader.num_workers = 1

    #### Generate Output directory
    #modrun = 'SEG_GEOPHYSICS_MOD_'+model+'_PD_'+str(pressure_mod)+'_RPT_'+str(rpt_test_mod)+'_NT_'+str(n_train)+'_NE_'+str(n_epochs)+'_ACT_'+activation+'_BATCH_'+str(batch_size)

    modrun = 'SEG_GEOPHYSICS_HMG_PA'

    batch_size = n_train
    n_epoch_stop = 50
    n_epoch_stop = 100

    outpath = os.path.join('./outputs',modrun)
    outInput = os.path.join(outpath,'TrainingData')
    outModels = os.path.join(outpath,'Models')
    
    if os.path.exists(outpath):
        print(outpath,' exists')
        print('\t ... overwriting!')
    else:
        os.makedirs(outInput)
        os.makedirs(outModels)
    
    outfile = os.path.join(outpath,'models_optims.pth')

    #### Generate Data
    # Use your own forward modeling algorithms to generate Training and Validations sets
    #x, y, x_t, y_t, x_scale, y_scale = YourDataLoaderOrGenerator()
    #                                   
    #                                   
    #### Save data
    # Here you probably want to save the training and validation sets 

    x_t = np.load(os.path.join(outInput,'X_Train.npy'))
    y_t = np.load(os.path.join(outInput,'Y_Train.npy'))
    x_t = torch.tensor(x_t, dtype=torch.float).to(device)
    y_t = torch.tensor(y_t, dtype=torch.float).to(device)

    x_v = x_t[n_train:,:]
    y_v = y_t[n_train:,:]
    x_v = torch.tensor(x_v, dtype=torch.float).to(device)
    y_v = torch.tensor(y_v, dtype=torch.float).to(device)

    dataset = utils.TensorDataset(x_t,y_t)      # create dataset
    dataloader = utils.DataLoader(dataset)      # create dataloader

    print('X\t',x_t.shape)
    print('Y\t',y_t.shape)

    #### Input/Output size
    D_in = x_t.shape[1]
    D_out = y_t.shape[1]
    HT = 1000               # Hidden Neurons
    m_par={                 # Size/Dropout/Decrease/Rates
        'mod_L_3d_Drop_Decrease' : [[HT, HT, HT],True,True,[0.0,0.3]]
    }

    models = {}
    optimizers = {}
    if idx_P_loss < 0:
        losses = np.empty((n_epochs,len(m_par)))
        accuracy = None
    else:
        losses = np.empty((n_epochs,len(m_par),2))
        accuracy = np.empty((n_epochs,len(m_par)))
    in_out_lr_relu = [D_in,D_out,learning_rate,activation]

    print('HDF batching / Batch size',hdf_batch,batch_size)

    # These training parameters were tested for optimal training
    lr = 0.0008
    wd = 0.000125

    models = {}
    models_early = {}
    optimizers = {}
    losses = np.empty((n_epochs,len(m_par)))
    accuracy =np.empty((n_epochs,len(m_par)))
    learn_rates = np.empty((n_epochs,len(m_par)))
    stopped = np.empty((n_epochs,len(m_par)))
    in_out_lr_relu = [D_in,D_out,lr,activation]

    for i,m in enumerate(m_par):
        print(m)
        tmp = mod.MLP2Regressor(D_in=D_in,H=m_par[m][0],D_out=D_out,
                            activation=activation,drop=m_par[m][1],
                            drop_decrease=m_par[m][2],droprates=m_par[m][3],
                            max_epoch=n_epochs, lr=lr, weight_decay=wd)
        _, loss, accu, lr_rate, stop = tmp.fit_accu_MSLR_EarlyStopSelect(x_t,y_t,x_v,y_v,batch_size=batch_size,
                            stop_nepoch=n_epoch_stop,schedule=False,stopping=early_stop,verbose=True)
        models[m] = tmp.model
        optimizers[m+'_optim'] = tmp.optimizer
        losses[:,i] = loss
        accuracy[:,i] = accu
        learn_rates[:,i] = lr_rate
        stopped[:,i] = stop

        if early_stop:
            models_early = models
            m_sd = torch.load('checkpoint.pt')
            models_early[m].load_state_dict(m_sd)


    util.save_all_models(outfile, m_par, models, optimizers, 
                    losses,x_scale,y_scale,in_out_lr_relu, accuracy=accuracy, 
                    learnrates=learn_rates,stop=stopped)
    util.save_all_models(outfile+'.early.pth', m_par, models_early, optimizers, 
                    losses,x_scale,y_scale,in_out_lr_relu, accuracy=accuracy, 
                    learnrates=learn_rates,stop=stopped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rock physics deep net')
    parser.add_argument('--model', type=str, default='m2ai', metavar='m2ai/biot/biot_mix',
                        help='RPT forward model M2AI (m2ai), Biot (biot), Biot Solid (biot_solid), or Biot_Mix (biot_mix) (default: m2ai)')
    parser.add_argument('--rpt_test_mod', type=int, default=0, metavar='N',
                        help='RPT configuration in forward model')
    parser.add_argument('--pressure_mod', type=int, default=1, metavar='N',
                        help='Pressure model Avseth (1) Lang&Grana (3) (default: 1)')
    parser.add_argument('--n_train', type=int, default=1000, metavar='N',
                        help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=500, metavar='N',
                        help='Number of validation samples')
    parser.add_argument('--n_epochs', type=int, default=10, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--activation', type=str, default='relu', metavar='relu OR tanh',
                        help='NN model activation (default: relu)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='N',
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--learn_cyclic', type=int, default=0, metavar='N',
                        help='LR scheduler One Cycle on (1) off (0) !!! requires batch training (--hdf_batch & --batch_size) !!! (default: 0)')
    parser.add_argument('--idx_P_loss', type=int, default=-1, metavar='N',
                        help='Additional pressure loss weighting - Pressure idx column (default: -1 (No added loss!))')
    parser.add_argument('--w_P', type=float, default=0.0, metavar='N',
                        help='w_P Pressure loss weighting (default: 0 (Only Sg weighted!))')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='Random seed (default: 1)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help='enables early stopping')
    parser.add_argument('--n_epoch_stop', type=int, default=30, metavar='N',
                        help='Number of epochs for early stopping (default: 30)')

    parser.add_argument("--batch_size", type=int, default=10, metavar='N', help="Training batch size (default: 10)")
    parser.add_argument('--hdf_batch', type=int, default=0, metavar='N', help='HDF5 batching on (1) off (0) (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='Muber of workers (default: 0)')

    args = parser.parse_args()

    if args.cuda:
        torch.backends.cudnn.deterministic = True

    main(args)
