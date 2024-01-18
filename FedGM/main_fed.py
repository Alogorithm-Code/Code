import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sys import exit
import os
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid_dirichlet, cifar10_iid,cifar10_noniid_dirichlet, cifar100_iid,cifar100_noniid_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate,fedprox_LocalUpdate,server_fedgm,server_finetune_fedgm,moon_LocalUpdate
from models.Update import server_train_fedmk, fedmk_local_update
from models.Nets import one_hot_embedding,MLP,SimpleCNN,Generator
from models.resnet import resnet18
from models.Fed import FedAvg
from models.test import test_img
from torch.autograd import Variable
import time

import os


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/MNIST', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/MNIST', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid == 'True':
            dict_users,count_lists = mnist_iid(dataset_train, args.num_users)
        elif args.iid == 'noniid_dirichlet':
            #exit('Error: only consider IID setting in CIFAR10')
            dict_users,count_lists = mnist_noniid_dirichlet(dataset_train, args.num_users, args.beta)
        else:
            pass

    elif args.dataset == 'cifar10':
        trans_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10)
        if args.iid == 'True':
            dict_users,count_lists = cifar10_iid(dataset_train, args.num_users)
        elif args.iid == 'noniid_dirichlet':
            #exit('Error: only consider IID setting in CIFAR10')
            dict_users,count_lists = cifar10_noniid_dirichlet(dataset_train, args.num_users, args.beta)
        else:
            pass

    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid == 'True':
            dict_users,count_lists = cifar100_iid(dataset_train, args.num_users)
        elif args.iid == 'noniid_dirichlet':
            #exit('Error: only consider IID setting in CIFAR10')
            dict_users,count_lists = cifar100_noniid_dirichlet(dataset_train, args.num_users, args.beta)
        else:
            pass

    else:
        exit('Error: unrecognized dataset')


    # build model

    if args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP().to(args.device)

    elif args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = SimpleCNN(num_classes=args.num_classes).to(args.device)
   
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = SimpleCNN(num_classes=args.num_classes).to(args.device)

    elif args.model == 'resnet18' and args.dataset == 'cifar100':
        net_glob = resnet18(num_classes=args.num_classes).to(args.device)

    else:
        exit('Error: unrecognized model')



    #store initial glob model for each client for moon
    if args.plan_mode == 'moon':
        for i in range(args.num_users):
            torch.save(net_glob.state_dict(),'./moon/dataset{}_iid{}_planmode{}_client{}_seed{}_model{}_beta{}.pth'.format(args.dataset,\
            args.iid,args.plan_mode,i,args.seed,args.model,args.beta))
          
    #net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    dummy_data_sets,dummy_label_sets, dummy_logit_sets = [], [], []
    #print('count_lists',np.array(count_lists).shape)

    # training
    test_acc_list, test_loss_list = [], []
    gain_acc_list = []
    #ensemble_weight_test_acc_list, ensemble_weight_test_loss_list = [], []
    

    for iteration in range(args.epochs):
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = list(np.random.choice(range(args.num_users), m, replace=False))
        print('{}th epoch select idxs_users{}'.format(iteration,idxs_users))
        start_time = time.time()
        for idx in idxs_users:
            if args.plan_mode == 'fedavg' or args.plan_mode == 'fedgm' or  args.plan_mode == 'fedftg':
                local = LocalUpdate(args=args,dataset=dataset_train, idxs=dict_users[idx], index=idx, testdataset=dataset_test)
                w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            elif args.plan_mode == 'moon':
                local = moon_LocalUpdate(args=args, index=idx, dataset=dataset_train, idxs=dict_users[idx])
                w = local.train(net=copy.deepcopy(net_glob).to(args.device))



            elif args.plan_mode == 'fedprox':
                local = fedprox_LocalUpdate(args=args,dataset=dataset_train, idxs=dict_users[idx], index=idx, testdataset=dataset_test)
                w = local.train(net=copy.deepcopy(net_glob).to(args.device))


            elif args.plan_mode == 'fedmk':
                if iteration  < 1:
                    fedmk_local = fedmk_local_update(args=args,dataset=dataset_train, idxs=dict_users[idx])
                    dummy_data_set, dummy_label_set = fedmk_local.fedmk_g(net=copy.deepcopy(net_glob).to(args.device))
                    dummy_data_sets.append(dummy_data_set)
                    dummy_label_sets.append(dummy_label_set)
                    #print(dummy_data_sets)
                else :
                    local = LocalUpdate(args=args,dataset=dataset_train, idxs=dict_users[idx], index=idx, testdataset=dataset_test)
                    w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            

            else:
                pass

            if  iteration  < 1:
                if args.plan_mode != 'fedmk' :
                    w_locals.append(copy.deepcopy(w))
            else :
                w_locals.append(copy.deepcopy(w))


        end_time = time.time()
        #print('local-train time',(end_time-start_time))

        

        # update global weights
        if iteration  < 200:
            if args.plan_mode != 'fedmk' :
                w_glob = FedAvg(w_locals)
        else :
            w_glob = FedAvg(w_locals)

        if args.plan_mode == 'fedavg' or args.plan_mode == 'fedprox' or args.plan_mode == 'moon' : 
            net_glob.load_state_dict(w_glob)           
            net_glob.eval()
            test_acc, test_loss = test_img(net_glob, dataset_test, args)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)


        elif args.plan_mode == 'fedgm':
            copy_net_glob = copy.deepcopy(net_glob).to(args.device)
            net_glob.load_state_dict(w_glob)
            test_acc, test_loss = test_img(net_glob, dataset_test, args)  
                
            if iteration < 10:
                s_fedgm = server_fedgm(args=args)
                if args.model == 'cnn' and args.dataset == 'cifar10':
                    local_model = SimpleCNN().to(args.device) 
                elif args.model == 'cnn' and args.dataset == 'cifar100':
                    local_model = SimpleCNN().to(args.device) 
                elif args.model == 'resnet18' and args.dataset == 'cifar100':
                    local_model = resnet18().to(args.device)
                elif args.model == 'mlp' and args.dataset == 'mnist':
                    local_model = MLP().to(args.device) 

                dummy_data_set, dummy_label_set, dummy_logit_set = s_fedgm.gm(net=copy_net_glob,local_model=local_model,idxs_users=idxs_users)
                dummy_data_sets.extend(dummy_data_set)
                dummy_label_sets.extend(dummy_label_set)
                dummy_logit_sets.extend(dummy_logit_set)
                if len(dummy_data_sets) > 10:
                    del dummy_data_sets[:10]
                    del dummy_label_sets[:10]
                    del dummy_logit_sets[:10]
                server_model = server_finetune_fedgm(args=args,dataset=dataset_test)

                new_w_glob,test_acc,test_loss = server_model.train(net=copy.deepcopy(net_glob).to(args.device), dummy_data_set=dummy_data_sets,dummy_label_set=dummy_label_sets,dummy_logit_set=dummy_logit_sets,lr_dyn=args.lr, iteration=iteration)

                net_glob.load_state_dict(new_w_glob)   

                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)

            else :
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)   
        


        elif args.plan_mode == 'fedftg':

            net_glob.load_state_dict(w_glob) 
            label_weights_ = []
            for i in range(len(idxs_users)):
                label_weights_.append(count_lists[idxs_users[i]])
            label_weights_ = np.array(label_weights_)#.T
            label_weights = []
            for i in range(m):
                label_weights.append(np.array(label_weights_[:,i]) / np.sum(label_weights_[:,i]))
            label_weights = np.array(label_weights)
            #print('label_weights: ',label_weights)

            if iteration < 200:

                server_generate = server_fedftg(args=args,testdataset=dataset_test)

                if args.model == 'mlp' and args.dataset == 'mnist':
                    local_model_list = [ MLP().to(args.device) for i in range(m)]

                elif args.model == 'cnn' and args.dataset == 'cifar10':
                    local_model_list = [ SimpleCNN(num_classes=args.num_classes).to(args.device) for i in range(m)]
               
                
                elif args.model == 'cnn' and args.dataset == 'cifar100':
                    local_model_list = [SimpleCNN(num_classes=args.num_classes).to(args.device) for i in range(m)]


                elif args.model == 'resnet18' and args.dataset == 'cifar100':
                    local_model_list = [resnet18(num_classes=args.num_classes).to(args.device) for i in range(m)]

                else:
                    exit('Error: unrecognized model')
                for i in range(m):
                    local_model_list[i].load_state_dict(torch.load('./client_model/dataset{}_iid{}_planmode{}_seed{}_clientidx{}_model{}_beta{}.pth'.format(args.dataset,args.iid,args.plan_mode,\
                                                                    args.seed,idxs_users[i],args.model,args.beta)))


                generator = Generator(nz=args.latent_dim, ngf=args.ngf, nc=args.num_channels, img_size=args.img_size, num_classes=args.num_classes).to(args.device)
                ###load existed generator
                if os.path.exists('./server_generator/dataset{}_iid{}_planmode{}_seed{}_model{}_beta{}.pth'.format(args.dataset,args.iid,args.plan_mode,args.seed,\
                                                                    args.model,args.beta)):
                    generator.load_state_dict(torch.load('./server_generator/dataset{}_iid{}_planmode{}_seed{}_model{}_beta{}.pth'.format(args.dataset,args.iid,args.plan_mode,args.seed,\
                                                                    args.model,args.beta)))

                #generator = Generator(latent_dim=args.latent_dim, ngf=args.ngf, channels=args.num_channels, num_classes=args.num_classes).to(args.device)
                new_w_glob,test_acc,test_loss = server_generate.train(net=copy.deepcopy(net_glob).to(args.device),local_nets=local_model_list,generator=copy.deepcopy(generator).to(args.device),label_weights=label_weights)
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)
                net_glob.load_state_dict(new_w_glob)
            else :
                test_acc, test_loss = test_img(net_glob, dataset_test, args) 
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)   
        

        elif args.plan_mode == 'fedmk':

            if iteration < 200:
                copy_net_glob = copy.deepcopy(net_glob).to(args.device)
                if len(dummy_data_sets) > 10:
                    del dummy_data_sets[:10]
                    del dummy_label_sets[:10]
                server_model = server_train_fedmk(args=args,dataset=dataset_test)
                new_w_glob,test_acc,test_loss = server_model.train(net=copy.deepcopy(net_glob).to(args.device), dummy_data_set=dummy_data_sets,dummy_label_set=dummy_label_sets,lr_dyn=lr_dyn, iteration=iteration)
                net_glob.load_state_dict(new_w_glob)    
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss) 

            else :
                net_glob.load_state_dict(w_glob)
                net_glob.eval()
                test_acc, test_loss = test_img(net_glob, dataset_test, args)
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)
                
        else:
            pass


    # plot loss curve
    plt.figure(figsize=(10,2))
    plt.subplots_adjust(wspace =0.4, hspace =0.4)#调整子图间距
    plt.subplot(121)
    plt.plot(range(len(test_acc_list)), test_acc_list)
    plt.ylabel('test_accuracy')
    plt.subplot(122)
    plt.plot(range(len(test_loss_list)), test_loss_list)
    plt.ylabel('test_loss')




    plt.savefig('./save/fed_{}_{}_C{}_iid{}_{}_seed{}_lr{}_weightdecray{}_beta{}_lambda{}_mu{}_alpha{}.png'.format(args.dataset, \
            args.epochs, args.frac, args.iid,args.plan_mode, args.seed, args.lr,args.reg,args.beta,args.Lambda,args.mu,args.alpha))

    result_dict = {'test_loss':test_loss_list,'test_accuracy':test_acc_list}

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv('./save/fed_{}_{}_C{}_iid{}_{}_seed{}_lr{}_weightdecray{}_beta{}_lambda{}_mu{}_alpha{}.csv'.format(args.dataset,\
            args.epochs, args.frac, args.iid,args.plan_mode, args.seed,args.lr,args.reg,args.beta,args.Lambda,args.mu,args.alpha),index=False,header=True)


    torch.save(net_glob.state_dict(),'./model_param/dataset{}_iid{}_planmode{}_seed{}_lr{}_weightdecray{}_beta{}_lambda{}_mu{}_alpha{}.pth'.format(args.dataset,\
            args.iid,args.plan_mode,args.seed,args.lr,args.reg,args.beta,args.Lambda,args.mu,args.alpha))
