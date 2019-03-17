from FM_utils import utils
from FM import FM_Model
import torch
import torch.utils.data as Data

X_train,Y_train,X_test,Y_test,features_num = utils()

#print(type(X_train))
device = 'cpu'
use_cuda = True

hidden_dim = [100,100,50]

if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda'

fm = FM_Model(X_train.shape[1],device,5,features_num,hidden_dim).to(device)

train_torch_data = Data.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train))
test_torch_data = Data.TensorDataset(torch.from_numpy(X_test),torch.from_numpy(Y_test))

fm.fit(train_torch_data,test_torch_data,120,0.03)