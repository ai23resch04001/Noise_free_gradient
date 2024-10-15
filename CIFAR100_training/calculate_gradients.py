import torch 
import argparse 
import warnings 
from utils import resume ,get_dataset,get_model
from sklearn.neighbors import NearestNeighbors
from functorch import make_functional_with_buffers
from torchmetrics.functional import pairwise_cosine_similarity
import pickle 
from torch.utils.data import DataLoader
import multiprocessing as mp
import numpy as np 
import time 
import os 
import subprocess
from tqdm import tqdm 


USE_TORCHMETRIC=1
process_start_time=time.time()
##disable warnings
warnings.warn("userwarning",UserWarning)
parser = argparse.ArgumentParser(description='PyTorch based Coreset selection')
parser.add_argument("--dataset",default='cifar10',type=str,help='chose cifar10,cifar100 or ilsvrc')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--model",default='resnet18',help='enter resnet18,vgg16,vit_small')
parser.add_argument("--patch",default=4,type=int,help="patch size for ViT model")
parser.add_argument("--dimhead",default=512,type=int)
parser.add_argument("--epochs",type=int,default=10,help="selection epochs")
parser.add_argument("--device",type=str,default="cuda:0",help="specific device id for training")
parser.add_argument("--radius",type=float,default=0.2,help="radius for selection of neighbors")
parser.add_argument("--corruption_percentage",type=int,default=5)
args = vars(parser.parse_args())

# function to compute gradients
def compute_loss_stateless_model(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = fmodel(params, buffers, batch)
    loss = criterion(predictions, targets)
    return loss


print("======================== Settings ========================")
print("Model:",args["model"]," ,dataset: ",args["dataset"]," ,batch size: ",args["batch_size"])
print("device: ",args["device"]," ,Selection Epochs: ",args["epochs"])
print("==========================================================")

#create the directory to store the neighbors calculation


args["dataset"]=args["dataset"].lower()
args["model"] = args["model"].lower()



if not os.path.exists("gradients_folder"):
    os.mkdir("gradients_folder")
if not os.path.exists("gradients_folder/{}".format(args["dataset"])):
    os.mkdir("gradients_folder/{}".format(args["dataset"]))
if not os.path.exists("gradients_folder/{}/{}".format(args["dataset"],args["model"])):
    os.mkdir("gradients_folder/{}/{}".format(args["dataset"],args["model"]))


#get the dataset
print("\n\n==> Loading the dataset....")
train_dataset = get_dataset(args,train=True,gradient_generation=False,gradient_calculation=True,shuffle=False) #will get the coreset
device = torch.device(args["device"])

if os.path.exists('mapping_data_{}.pkl'.format(args["dataset"])):
    f=open('mapping_data_{}.pkl'.format(args["dataset"]),"rb")
    mapping_dict = pickle.load(f)
else:
    print("[INFO] mapping data does not exist. generating...")
    subprocess.call(['python3','../map_images.py','--dataset',args["dataset"]])
    f=open('mapping_data_{}.pkl'.format(args["dataset"]),"rb")
    mapping_dict = pickle.load(f)


print("\n==> Running the gradient calculation loop...")
# for epoch in range(args["se"]):
if True:
    # epoch = args["se"]
    for epoch in range(args["epochs"]):
        print("\n-------------------Epoch:{}-------------------\n".format(epoch+1))
        print("\n==> Creating the model...")
        model = get_model(args)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        if args["dataset"]=="cifar100c":
            print("[INFO] loading model : checkpoint/{}/{}_{}_percent/saved_model_{}.pth".format(args["dataset"],args["model"],args["corruption_percentage"],epoch))
            resume(model,"checkpoint/{}/{}_{}_percent/saved_model_{}.pth".format(args["dataset"],args["model"],args["corruption_percentage"],epoch))
        else:
            print("[INFO] loading model: checkpoint/{}/{}/saved_model_{}.pth".format(args["dataset"],args["model"],epoch))
            resume(model,"checkpoint/{}/{}/saved_model_{}.pth".format(args["dataset"],args["model"],epoch))
        print("\n==> Gradient calculation...")

        model.eval()
        fmodel, params, buffers = make_functional_with_buffers(model)
        for param in params:
            param.requires_grad_(False)

        ft_compute_grad = torch.func.grad(compute_loss_stateless_model)
        ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        pbar2 = tqdm(range(len(mapping_dict.keys())), position=0)
        count2 = 0
        process_counter = 0
        if args['dataset']=='cifar10':
            process_num = 10
        elif args['dataset']=='cifar100':
            process_num = 10
        elif args["dataset"]=='cifar100c':
            process_num = 25
        else:
            process_num = 40
        
        
        process_num = 2
        for data_class in mapping_dict.keys():
            if data_class % process_num == 0:
                gradient_dict = {}
            train_subset = torch.utils.data.Subset(train_dataset, mapping_dict[data_class])
            train_subset_loader = DataLoader(
                train_subset, batch_size=args["batch_size"], num_workers=8, shuffle=False
            )
            gradient_list = []

            if args["dataset"]=='cifar100c':
                for _,data, targets in train_subset_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
                    #print(ft_per_sample_grads[-2].shape)
                    if len(gradient_list) == 0:
                        if args["model"]=="resnet50" or args["model"]=="inception":

                            if args["dataset"]=='cifar10':
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 20480)
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100c":
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 204800)
                            else:
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 128000)

                        else:
                            if args["dataset"]=='cifar10':
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 5120)
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100c":
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 51200)
                            else:
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 128000)

                    else:
                        if args["model"]=="resnet50" or args["model"]=="inception":
                            if args["dataset"]=='cifar10':
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 20480)))
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100c":
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 204800)))
                            else:
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 128000)))

                        else:
                            if args["dataset"]=='cifar10':
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 5120)))
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100c":
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 51200)))
                            else:
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 128000)))

            else:
                for data, targets in train_subset_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
                    #print(ft_per_sample_grads[-2].shape)
                    if len(gradient_list) == 0:
                        if args["model"]=="resnet50" or args["model"]=="inception":

                            if args["dataset"]=='cifar10':
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 20480)
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100n":
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 204800)
                            else:
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 128000)

                        elif args["model"]=="vgg":
                            if args["dataset"]=='cifar10':
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 40960)
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100n":
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 409600)
                            else:
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 128000) 
                        else:
                            if args["dataset"]=='cifar10':
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 5120)
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100n":
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 51200)
                            else:
                                gradient_list = ft_per_sample_grads[-2].reshape(-1, 128000)

                    else:
                        if args["model"]=="resnet50" or args["model"]=="inception":
                            if args["dataset"]=='cifar10':
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 20480)))
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100n":
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 204800)))
                            else:
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 128000)))
                        elif args["model"]=="vgg":
                            if args["dataset"]=='cifar10':
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 40960)))
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100n":
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 409600)))
                            else:
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 128000)))
                        else:
                            if args["dataset"]=='cifar10':
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 5120)))
                            elif args["dataset"]=='cifar100' or args["dataset"]=="cifar100n":
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 51200)))
                            else:
                                gradient_list = torch.vstack((gradient_list ,ft_per_sample_grads[-2].reshape(-1, 128000)))


            samples = gradient_list.cpu()
            samples_distances = torch.clamp(1-pairwise_cosine_similarity(samples),min=0)
            nbrs=NearestNeighbors(radius=args["radius"],algorithm='auto',n_jobs=-1,metric='precomputed').fit(samples_distances)
            distances,indices = nbrs.radius_neighbors(samples_distances)
            with open("gradients_folder/{}/{}/neighbors_{}_{}.pkl".format(args["dataset"],args["model"],data_class,epoch), "wb") as f:
                pickle.dump(distances, f) 

            pbar2.update()
            del train_subset_loader,gradient_list,samples,train_subset

print("------------------------------------------")
print("[INFO] Total time taken:{:.3f} seconds".format(time.time()-process_start_time))
