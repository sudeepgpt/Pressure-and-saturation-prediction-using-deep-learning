import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
import torch

n_train = 200000
n_valid = 50000

out_path = './outputs/SEG_GEOPHYSICS_biot_Pressure_Avseth/'

# Input data
# _o: Original unscaled
# _s: Scaled 
pth_dat_x_o = out_path + 'TrainingData/X_Train_Original.npy'
pth_dat_x_s = out_path + 'TrainingData/X_Train.npy'

pth_dat_y_o = out_path + 'TrainingData/Y_Train_Original.npy'
pth_dat_y_s = out_path + 'TrainingData/Y_Train.npy'

# Load Original
# This is real valued poro-elastic attributes
x_test = np.load(pth_dat_x_o)
y_test = np.load(pth_dat_y_o)

'''
-------------------------------
X - Data consists of 12 columns
-------------------------------
- Vp, Vs, rho, Qp, Qs @ Time 0
- Vp, Vs, rho, Qp, Qs @ Time 1
- Pressure P0 @ Time 0
- Depth 
-------------------------------
Y - Data consists of 4 columns
-------------------------------
- Porosity
- Sg 
- Pressure
- Vcl
'''

x_test = x_test[:n_train,:]
y_test = y_test[:n_train,:]

# Load scaled 
# This is scaled poro-elastic attributes
x_test_t_should_Match_Below = np.load(pth_dat_x_s)
y_test_t_should_Match_Below = np.load(pth_dat_y_s)

print(x_test.shape,x_test_t_should_Match_Below.shape)
print(y_test.shape,y_test_t_should_Match_Below.shape)

###################################################
# Scaling of inputs
###################################################

# Scaler paths
pth_scale_label     = out_path + 'scaler_label.pkl'
pth_scale_feature   = out_path + 'scaler_feature.pkl'

# Load scalers
y_scaler = joblib.load(pth_scale_feature)
x_scaler = joblib.load(pth_scale_label)

# Convert input values with scalers
x_test_t = torch.from_numpy(x_scaler.transform(x_test)).float()
y_test_t = torch.from_numpy(y_scaler.transform(y_test)).float()

# Not matching as there are rounding errors
print(np.array_equal(x_test_t_should_Match_Below,x_test_t))
print(x_test[0:5,:])
print(x_test_t[0:5,:])
print(x_test_t_should_Match_Below[0:5,:])

# Make predictions with scalers
# predictions should be made with the Poro-elastic attributes

predictions = y_test_t
predictions = y_scaler.inverse_transform(predictions.detach().numpy())
print('################################')
print(predictions.shape)
print(predictions)