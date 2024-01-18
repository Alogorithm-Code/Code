import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', type=str, default='False', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', type=str, default='False', help='aggregation over all clients')
    parser.add_argument('--plan_mode', type=str, default='original', help='train scheme')

    #local client generate noise data
    parser.add_argument('--g_epoch', type=int, default=200, help='each client or server train a generator')
    parser.add_argument('--s_epoch', type=int, default=20, help='server local train epoch based generator')
    parser.add_argument('--g_bs', type=int, default=500, help='batch size for generator')
    parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
    parser.add_argument('--ngf', type=int, default=64, help='generator hidden variable dimensionality ')
    parser.add_argument('--fc_dim', type=int, default=200, help='(200, 512, 512)for simplecnn mlp resnet18 ')

    parser.add_argument('--oh', type=float, default=0.05, help='one hot loss')
    parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
    parser.add_argument('--a', type=float, default=0.01, help='activation loss')

    parser.add_argument('--lr_G', type=float, default=0.01 , help='generator learning rate')
    parser.add_argument('--img_size', type=int, default=32 , help='generator fig size') 
    parser.add_argument('--pseudo_data_number', type=int, default=500 , help='pseudo_data_number')
    parser.add_argument('--reg', type=float, default=1e-5, help='optimizer weight decray strength for task model')
    parser.add_argument('--reg_G', type=float, default=1e-5, help='optimizer weight decray strength for generator model')




    parser.add_argument('--Lambda', type=float, default=1.0, help='pseudo label for server finetune')
    parser.add_argument('--mu', type=float, default=1.0, help='self-label')
    parser.add_argument('--beta', type=float, default=0.5, help='data noniid dirichlet factor')

    parser.add_argument('--alpha', type=float, default=0.05, help='loss proximate factor of fedprox / moon')
    parser.add_argument('--temp', type=float, default=2, help='distillation factor for fedmoon')


    
    args = parser.parse_args()
    return args
