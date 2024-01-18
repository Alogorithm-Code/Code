import torch
from torch import nn, autograd, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
import os
import json
import os
from torch.autograd import Variable
from models.Nets import one_hot_embedding
from torchvision.utils import save_image
import time
from models.reparam_module import ReparamModule

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def cross_entropy_soft(input,target, reduction='mean'):
    logprobs = F.log_softmax(input,dim=-1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

class LocalUpdate(object):
    def __init__(self, args,index, dataset=None, idxs=None, testdataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.index = index
        self.testdataset = DataLoader(testdataset, batch_size=self.args.bs)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.reg)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                probs, feature = net(images)
                loss = self.loss_func(probs, labels)
                loss.backward()
                optimizer.step()

        torch.save(net.state_dict(),'./client_model/dataset{}_iid{}_planmode{}_seed{}_clientidx{}_model{}_beta{}.pth'.format(self.args.dataset,self.args.iid,\
                                        self.args.plan_mode,self.args.seed,self.index,self.args.model,self.args.beta))
        return net.state_dict()



class fedprox_LocalUpdate(object):
    def __init__(self, args,index, dataset=None, idxs=None, testdataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.index = index
        self.testdataset = DataLoader(testdataset, batch_size=self.args.bs)

    def train(self, net):
        global_net = copy.deepcopy(net).to(self.args.device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.reg)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                probs, feature = net(images)
                proximal_term = 0.0
                for lw, gw in zip(net.parameters(),global_net.parameters()):
                	proximal_term += (lw - gw).norm(2)

                loss = self.loss_func(probs, labels) + self.args.alpha * proximal_term
                loss.backward()
                optimizer.step()

        torch.save(net.state_dict(),'./client_model/dataset{}_iid{}_planmode{}_seed{}_clientidx{}_model{}_beta{}.pth'.format(self.args.dataset,self.args.iid,\
                                        self.args.plan_mode,self.args.seed,self.index,self.args.model,self.args.beta))

        return net.state_dict()



class server_fedgm(object):
    def __init__(self, args):
        self.args = args

    def gm(self,net,local_model,idxs_users):
        dummy_data_set = []
        dummy_label_set = []
        dummy_logit_set = []
        start_time = time.time()

        for i in range(len(idxs_users)):
            local_model.load_state_dict(torch.load('./client_model/dataset{}_iid{}_planmode{}_seed{}_clientidx{}_model{}_beta{}.pth'.format(self.args.dataset,self.args.iid,\
                                        self.args.plan_mode,self.args.seed,idxs_users[i],self.args.model,self.args.beta)))
            dy_dx = []
            for lw, gw in zip(net.parameters(),local_model.parameters()):
                t = ((lw - gw).detach()) 
                dy_dx.append(t)
            original_dy_dx = list(dy_dx)

            dummy_data = torch.randn((50,self.args.num_channels,self.args.img_size,self.args.img_size)).to(self.args.device).requires_grad_(True)
            dummy_label = torch.randn((50,self.args.num_classes)).to(self.args.device).requires_grad_(True)
            optimizer_G = torch.optim.Adam([dummy_data, dummy_label], lr=0.1, weight_decay=1e-5)
            
            for epoch in range(self.args.g_epoch):
                optimizer_G.zero_grad()
                outputs_T, features_T = net(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=1)
                dummy_loss = cross_entropy_soft(outputs_T,dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                    grad_diff += ((1 - torch.nn.functional.cosine_similarity(gx.flatten(),gy.flatten(),0,1e-8)) + 1*((gx - gy) ** 2).sum())
                grad_diff.backward()
                optimizer_G.step()
            dummy_logit, _ = local_model(dummy_data.detach())


            dummy_data_set.append(dummy_data.detach())
            dummy_label_set.append(dummy_label.detach())
            dummy_logit_set.append(dummy_logit.detach())

        return dummy_data_set,dummy_label_set,dummy_logit_set



class server_finetune_fedgm(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.test = DataLoader(dataset, batch_size=self.args.bs)

    def train(self, net,dummy_data_set,dummy_label_set,dummy_logit_set,lr_dyn, iteration):
        net.train()
        dummy_dataset = torch.stack(dummy_data_set).view(500,self.args.num_channels,self.args.img_size,self.args.img_size)
        dummy_label = torch.stack(dummy_label_set).view(500,self.args.num_classes)
        dummy_logit = torch.stack(dummy_logit_set).view(500,self.args.num_classes)

        length = dummy_dataset.size()[0]
        shuffle_idx = np.random.permutation(np.arange(length))
        dummy_dataset = dummy_dataset[shuffle_idx].view(500,self.args.num_channels,self.args.img_size,self.args.img_size)
        dummy_label = dummy_label[shuffle_idx].view(500,self.args.num_classes)
        dummy_logit = dummy_logit[shuffle_idx].view(500,self.args.num_classes)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr_dyn,  momentum=self.args.momentum, weight_decay=self.args.reg)  
        start_time = time.time()
        for epoch in range(self.args.s_epoch):
            net.train()
            for j in range(int(len(dummy_dataset)/50)):
                optimizer.zero_grad()
                outputs_T, _ = net(dummy_dataset[j*50:(j+1)*50])
                dummy_onehot_label = F.softmax(dummy_label[j*50:(j+1)*50], dim=1)
                dummy_onehot_label1 = F.softmax(dummy_logit[j*50:(j+1)*50], dim=1)
                loss = self.args.Lambda *  cross_entropy_soft(outputs_T,dummy_onehot_label) + self.args.mu * cross_entropy_soft(outputs_T,dummy_onehot_label1)
                loss.backward()
                optimizer.step() 


        return copy.deepcopy(net.state_dict()),test_accuracy,test_loss


class fedmk_local_update(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()

    def fedmk_g(self,net):
        for i in range(1):
            dummy_data = torch.randn((50,self.args.num_channels,self.args.img_size,self.args.img_size)).to(self.args.device).requires_grad_(True)
            dummy_label = torch.randn((50,self.args.num_classes)).to(self.args.device).requires_grad_(True)
            optimizer_G = torch.optim.Adam([dummy_data, dummy_label], lr=0.1, weight_decay=1e-5)
            
            target_params = [torch.cat([p.data.to(self.args.device).reshape(-1) for p in net.parameters()], 0).requires_grad_(True)]
            net = ReparamModule(net)
            
            for epoch in range(self.args.g_epoch):
                optimizer_G.zero_grad()
                outputs_T, features_T = net(dummy_data,flat_param=target_params[-1])
                dummy_onehot_label = F.softmax(dummy_label, dim=1)
                dummy_loss = cross_entropy_soft(outputs_T,dummy_onehot_label)
                grad = torch.autograd.grad(dummy_loss, target_params[-1], create_graph=True)[0]
                train_param = target_params[-1] - self.args.lr * grad
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    probs, feature = net(images,flat_param=train_param)
                    loss = self.loss_func(probs, labels)
                    loss.backward()
                    optimizer_G.step()             


        return dummy_data.detach(),dummy_label.detach()

class server_train_fedmk(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.test = DataLoader(dataset, batch_size=self.args.bs)

    def train(self, net,dummy_data_set,dummy_label_set,lr_dyn, iteration):
        net.train()
        dummy_dataset = torch.stack(dummy_data_set).view(500,self.args.num_channels,self.args.img_size,self.args.img_size)
        dummy_label = torch.stack(dummy_label_set).view(500,self.args.num_classes)


        length = dummy_dataset.size()[0]
        shuffle_idx = np.random.permutation(np.arange(length))
        dummy_dataset = dummy_dataset[shuffle_idx].view(500,self.args.num_channels,self.args.img_size,self.args.img_size)
        dummy_label = dummy_label[shuffle_idx].view(500,self.args.num_classes)
       
        optimizer = torch.optim.SGD(net.parameters(), lr=lr_dyn,  momentum=self.args.momentum, weight_decay=self.args.reg)  
        for epoch in range(self.args.s_epoch):
            net.train()
            for j in range(int(len(dummy_dataset)/50)):
                optimizer.zero_grad()
                outputs_T, _ = net(dummy_dataset[j*50:(j+1)*50])
                dummy_onehot_label = F.softmax(dummy_label[j*50:(j+1)*50], dim=1)
                loss = self.args.gl * cross_entropy_soft(outputs_T,dummy_onehot_label) 
                loss.backward()
                optimizer.step()   
            if (epoch+1) % 1 == 0:
                net.eval()
                total_correct = 0
                test_loss = 0
                for batch_idx, (images, labels) in enumerate(self.test):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    probs, feature = net(images)
                    pred = probs.data.max(1)[1]
                    test_loss += F.cross_entropy(probs, labels).item()
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()

                test_loss = test_loss*self.args.bs/len(self.test.dataset)
                test_accuracy = 100*float(total_correct) / len(self.test.dataset)
                
                print('%d-th server retrain Epoch Test Accuracy: %f' % (epoch+1, 100*float(total_correct) / len(self.test.dataset))) 
            

        return copy.deepcopy(net.state_dict()),test_accuracy,test_loss




class server_fedftg(object):
    def __init__(self, args, testdataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.test = DataLoader(testdataset, batch_size=self.args.bs)

    def train(self, net, local_nets, generator, label_weights):


        optimizer_GM = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
        optimizer_Generator = torch.optim.Adam(generator.parameters(), lr=self.args.lr_G, weight_decay=self.args.reg_G)

        for epoch in range(self.args.g_epoch):
            for i in range(self.args.s_epoch):
                net.eval()
                generator.train()

                z1 = Variable(torch.randn(int(self.args.g_bs/2), self.args.latent_dim)).to(self.args.device)
                z2 = Variable(torch.randn(int(self.args.g_bs/2), self.args.latent_dim)).to(self.args.device)
                z = torch.cat((z1, z2), 0)
                label = np.random.randint(0,self.args.num_classes,self.args.g_bs)
                n_label = Variable(torch.LongTensor(label)).to(self.args.device)
                noise_label = one_hot_embedding(labels=n_label,num_classes=self.args.num_classes).to(self.args.device)
                optimizer_Generator.zero_grad()
                gen_imgs = generator(z,noise_label)
                gen_imgs1, gen_imgs2 = torch.split(gen_imgs, z1.size(0), dim=0)
                #diversity_loss = diversity_loss(eps, gen_output) 
                local_loss = 0
                ensemble_logit = 0
                #activation_loss = 0
                for i in range(len(local_nets)):
                    local_nets[i].eval()
                    weight = label_weights[:,i]
                    expand_weight = np.tile(weight, (self.args.g_bs,1))
                    outputs_T, features_T = local_nets[i](gen_imgs)
                    #local_predict_loss = torch.mean( self.loss_func(outputs_T, n_label) * (torch.tensor(weight, dtype=torch.float32).to(self.args.device)))
                    #local_loss += local_predict_loss
                    ensemble_logit+= outputs_T * (torch.tensor(expand_weight, dtype=torch.float32).to(self.args.device))
                outputs_T_global, features_T_global = net(gen_imgs)
                dummy_ensemble_logit = F.softmax(ensemble_logit, dim=1)
                kl_loss = cross_entropy_soft(outputs_T_global,dummy_ensemble_logit)
                local_loss = self.loss_func(ensemble_logit, n_label)

                ###计算diversity loss
                #ensemble_logit1, ensemble_logit2 = torch.split(ensemble_logit, z1.size(0), dim=0)
                #softmax_ensemble_logit1 = torch.nn.functional.softmax(ensemble_logit1, dim=1)
                #softmax_ensemble_logit2 = torch.nn.functional.softmax(ensemble_logit2, dim=1)
                #lz = torch.norm(gen_imgs2 - gen_imgs1) / torch.norm(softmax_ensemble_logit2 - softmax_ensemble_logit1)
                lz = torch.norm(gen_imgs2 - gen_imgs1) / torch.norm(z2 - z1)
                diversity_loss = 1 / (lz + 1 * 1e-20)
                loss = 2*local_loss + self.args.dl*diversity_loss - self.args.alpha*kl_loss 


                loss.backward()
                optimizer_Generator.step()


 
            net.train()
            generator.eval()
            dummy_dataset = []
            dummy_output = []
            dummy_label = []


            z = Variable(torch.randn(500, self.args.latent_dim)).to(self.args.device)
            label = np.random.randint(0,self.args.num_classes,500)
            n_label = Variable(torch.LongTensor(label)).to(self.args.device)
            noise_label = one_hot_embedding(n_label).to(self.args.device)
            gen_imgs = generator(z,noise_label)
            ensemble_logit = 0
            for i in range(len(local_nets)):
                local_nets[i].eval()
                weight = label_weights[:,i]
                expand_weight = np.tile(weight, (self.args.g_bs,1))
                outputs_T, features_T = local_nets[i](gen_imgs)
                ensemble_logit+= outputs_T* (torch.tensor(expand_weight, dtype=torch.float32).to(self.args.device))
            dummy_dataset.append(gen_imgs.detach())
            dummy_label.append(n_label.detach())
            dummy_output.append(ensemble_logit.detach())

            dummy_dataset = torch.stack(dummy_dataset).view(-1,self.args.num_channels,self.args.img_size,self.args.img_size)
            dummy_label = torch.stack(dummy_label).view(-1)
            dummy_output = torch.stack(dummy_output).view(-1,self.args.num_classes)
            length = dummy_dataset.size()[0]
            shuffle_idx = np.random.permutation(np.arange(length))
            dummy_dataset = dummy_dataset[shuffle_idx].view(500,self.args.num_channels,self.args.img_size,self.args.img_size)
            dummy_label = dummy_label[shuffle_idx].view(500)
            dummy_output = dummy_output[shuffle_idx].view(500,self.args.num_classes)

            for j in range(5):
                for idx in range(int(len(dummy_dataset)/50)):
                    optimizer_GM.zero_grad()
                    outputs_T, _ = net(dummy_dataset[idx*50:(idx+1)*50])
                    dummy_onehot_label = F.softmax(dummy_output[idx*50:(idx+1)*50], dim=1)
                    loss = cross_entropy_soft(outputs_T,dummy_onehot_label) 
                    #print(loss)
                    loss.backward()
                    optimizer_GM.step()   
            if (epoch+1) % 1 == 0:
                #net.eval()
                total_correct = 0
                test_loss = 0
                for batch_idx, (images, labels) in enumerate(self.test):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    probs, feature = net(images)
                    pred = probs.data.max(1)[1]
                    test_loss += F.cross_entropy(probs, labels).item()
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()

                test_loss = test_loss*self.args.bs/len(self.test.dataset)
                test_accuracy = 100*float(total_correct) / len(self.test.dataset)
                
                print('%d-th server retrain Epoch Test Accuracy: %f' % (epoch+1, 100*float(total_correct) / len(self.test.dataset))) 


        torch.save(generator.state_dict(),'./server_generator/dataset{}_iid{}_planmode{}_seed{}_model{}_beta{}.pth'.format(self.args.dataset,self.args.iid,self.args.plan_mode,\
                                    self.args.seed,self.args.model,self.args.beta))


        return copy.deepcopy(net.state_dict()),test_accuracy,test_loss





class moon_LocalUpdate(object):
    def __init__(self, args, index, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.index = index 

    def train(self, net):
        #net.train()
        glob_net = copy.deepcopy(net).to(self.args.device)
        previous_net = copy.deepcopy(net).to(self.args.device)
        previous_net.load_state_dict(torch.load('./moon/dataset{}_iid{}_planmode{}_client{}_seed{}_model{}_beta{}.pth'.format(self.args.dataset,\
            self.args.iid,self.args.plan_mode,self.index,self.args.seed,self.args.model,self.args.beta)))


        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        #optimizer = optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.reg)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                label_index = (labels.reshape(-1)).numpy()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                probs, feature = net(images)
                glob_probs, glob_feature = glob_net(images)
                previous_probs, previous_feature = previous_net(images)

                cos = torch.nn.CosineSimilarity(dim=-1)
                pos_cos = cos(feature, glob_feature).reshape(-1,1)
                neg_cos = cos(feature, previous_feature).reshape(-1,1)

                target = torch.zeros(images.size(0)).long().to(self.args.device)
                com_cos = torch.cat([pos_cos,neg_cos],dim=1)
                com_cos /= self.args.temp
                com_loss = self.loss_func(com_cos, target)

                ce_loss = self.loss_func(probs, labels)
                loss = ce_loss + self.args.mu*com_loss

                loss.backward()
                optimizer.step()


        torch.save(net.state_dict(), './moon/dataset{}_iid{}_planmode{}_client{}_seed{}_model{}_beta{}_alpha{}.pth'.format(self.args.dataset,\
            self.args.iid,self.args.plan_mode,self.index,self.args.seed,self.args.model,self.args.beta))

        return net.state_dict()
