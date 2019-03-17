
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FM_Model(nn.Module):

    def __init__(self,
                 emb_len,
                 device,
                 k,
                 features_num,
                 hidden_dim):
        super(FM_Model,self).__init__()

        self.device = device
        self.k = k
        self.emb_len = emb_len
        self.features_num = features_num
        self.features_num_len = len(features_num)
        self.hiddend_dim = hidden_dim
        self.hiddend_dim_len = len(hidden_dim)
        '''
        FM part
        '''
        self.first_linear = nn.Linear(self.emb_len,1)
        self.second_order_emb_list = nn.ModuleList(
            [nn.Embedding(num, self.k) for num in self.features_num]
        )

        '''
        DeepFM part
        '''

        self.deep_linear_first = nn.Linear(self.emb_len * self.k,self.hiddend_dim[0])
        self.deep_linear_list = nn.ModuleList(
            [nn.Linear(self.hiddend_dim[dim], self.hiddend_dim[dim+1]) for dim in range(1,self.hiddend_dim_len-1)]
        )

        self.deep_linear_last = nn.Linear(self.hiddend_dim[self.hiddend_dim_len-1],1)

        self.out = nn.Sigmoid()

    def forward(self,X):
        ''''''
        '''
        FM
        '''
        first_order = self.first_linear(X)

        first_index = 0
        second_index = 0

        temp = []

        for index,num in enumerate(self.features_num):
            second_index += num
            a = self.second_order_emb_list[index](X[:,first_index:second_index].long())
            #a = [(torch.sum(a[:,index,:],1)).unsqueeze(1) for index in range(num)]
            #temp.append(torch.cat(a,1))
            #print(a.size())
            temp.append(a)
            first_index = second_index

        '''
        this is feature intersaction without one-hot encoding   
        #temp = [torch.sum(self.second_order_emb_list[col](X[:,col].long()),1).unsqueeze(1) for col in range(self.emb_len)]
        '''
        emb_all = torch.cat(temp,1)

        v = torch.sum(emb_all,-1)
        #print(v.size(),X.size())
        item = v * X
        #print(item)

        second_sum_square = pow(torch.sum(item,1) , 2 )
        second_square_sum = torch.sum(item*item,1)
        second_order =  (0.5 * (second_sum_square - second_square_sum)).unsqueeze(1)

        fm_part = first_order + second_order

        '''
        Deep part
        '''

        x_deep = self.deep_linear_first(emb_all.view(emb_all.size(0),-1))
        x_deep = F.relu(x_deep)
        x_deep = F.dropout(x_deep,0.5)

        for each_linear in self.deep_linear_list:
            x_deep = each_linear(x_deep)
            x_deep = F.relu(x_deep)
            x_deep = F.dropout(x_deep,0.5)
        x_deep = self.deep_linear_last(x_deep)
        deep_part = x_deep

        '''
        output
        '''
        y_pred = self.out(fm_part + deep_part)

        return y_pred

    def fit(self,trainset,testset,n_epoch,lr):

        print(self.device)
        train_loader = DataLoader(dataset=trainset,shuffle=True,batch_size=100)

        net = self.train()
        loss_func = F.binary_cross_entropy
        optim = torch.optim.SGD(self.parameters(),lr=lr)


        print('linear with intersaction')
        print('='*50)
        for epoch in range(n_epoch):

            loss_epoch = 0

            #if abs(loss_last - loss_now) < 0.0
            for index , (x_train,y_train) in enumerate(train_loader):


                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()

                y_pred = net(x).squeeze()
                #print(y_pred)
                optim.zero_grad()
                loss = loss_func(y_pred,y)

                loss_epoch += loss
                loss.backward()
                optim.step()

                if index == len(train_loader) - 1:
                    t = ((y_pred > 0.5).float() == y).sum().float() / len(y)
                    print('accuracy : %.2f%%' % (t.cpu().numpy() * 100))
            print('epoch : {0} ,Loss : {1}'.format(epoch,loss_epoch))

        self.test(testset,net)

    def test(self,testloader,net):

        net.eval()
        test_loader = DataLoader(dataset=testloader,shuffle=True,batch_size=100)

        num_correct , num_sample = 0 , 0

        with torch.no_grad():
            for index , (x_test,y_test) in enumerate(test_loader):

                x = x_test.to(self.device).float()
                y = y_test.to(self.device).float()

                #print(x.size(),y.size())

                y_pred = net(x).squeeze()

                num_correct += ((y_pred > 0.5).float() == y).sum().float()
                num_sample += y.size(-1)


            accuracy =  num_correct / num_sample
            print('='*50)
            print('test accuracy : %.2f%%' % (accuracy.cpu().numpy() * 100))