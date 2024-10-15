#imports 
from tqdm import tqdm 
import argparse 
import warnings 
import random
import torch 
import time 
import numpy as np 
import wandb
import subprocess 
from utils import get_model,get_dataset,checkpoint,resume,train_model,train_model_perturbed,get_optimizer_criterion,test_model
import csv 
import os 
from datetime import datetime 
##disable warnings
warnings.warn("userwarning",UserWarning)
parser = argparse.ArgumentParser(description='PyTorch based Coreset selection')
parser.add_argument("--epochs",type=int,default=10,help='number of epochs for initial selection')
parser.add_argument("--dataset",default='cifar10',type=str,help='chose cifar10,cifar100 or ilsvrc')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--radius",type=float,default=0.2)
parser.add_argument("--multi_gpu",type=int,default=0)
parser.add_argument("--model",default='resnet18',help='enter resnet18,vgg16,vit_small')
parser.add_argument("--patch",default=4,type=int,help="patch size for ViT model")
parser.add_argument("--dimhead",default=512,type=int)
parser.add_argument("--se",type=int,default=10,help="selection epochs")
parser.add_argument("--num_gpus",type=int,default=1)
parser.add_argument("--device",type=str,default="cuda:0",help="specific device id for training")
parser.add_argument("--lr",type=float,default=0.1,help="learning rate for SGD optimizer")
parser.add_argument("--weight_decay",type=float,default=0.0001,help="learning rate for SGD optimizer")
parser.add_argument("--fraction",type=float,default=0.1,help="size of the coreset to be selected")
parser.add_argument("--optimizer",default="SGD",help="Enter SGD or Adam")
parser.add_argument("--min_lr",type=float,default=0.0001,help="minimum LR for scheduler")
parser.add_argument("--scheduler",default="cosine",help="Enter cosine or reducelr")
parser.add_argument("--iterations",type=int,default=1)
parser.add_argument("--random_seed",type=int,default=1)
parser.add_argument("--test_interval",type=int,default=1)
parser.add_argument("--corruption_percentage",type=int,default=5)
parser.add_argument("--test_eval_start_epoch",type=int,default=100,help="epochs after which to start test set evaluation,usually to speed up the training process")
parser.add_argument("--wandb",type=int,default=0)
parser.add_argument("--progress_bar",type=int,default=0)
parser.add_argument("--run_name",default=None)
args = vars(parser.parse_args())



args["dataset"]=args["dataset"].lower()
args["model"] = args["model"].lower()
#initialize wandb run
if args["wandb"]==1:
    if args["run_name"] is not None:
        name="{}_{}_{}_{}".format(args["model"],args["dataset"],args["fraction"],args["run_name"])
    else:
        name="{}_{}_{}".format(args["model"],args["dataset"],args["fraction"])
    wandb.init(
        project="{} training with {}".format(args["dataset"],args["model"]),
        name=name
    )
#get the dataset
print("\n\n==> Loading the dataset....")
train_loader,num_images = get_dataset(args,train=True,gradient_generation=False) #will get the coreset 
test_loader =  get_dataset(args,train=False,gradient_generation=False)


device = torch.device(args["device"])
if args["multi_gpu"]==0:
    NUM_GPUS=1
else:
    NUM_GPUS=args["num_gpus"]
if args["scheduler"]=="cosine":
    apply_scheduler=True
else:
    apply_scheduler=False 


#training loop
print("\n==> Starting the training loop...\n")
best_test_acc_list = []
for iteration in range(args["iterations"]):
    print("\n ---------------------- Iteration : {} / {} ------------------\n".format(iteration+1,args["iterations"]))
    print("======================== Settings ========================")
    print("Model:",args["model"],", dataset: ",args["dataset"],", batch size: ",args["batch_size"],", fraction: ",args["fraction"], ", radius: ",args["radius"])
    print("device: ",args["device"],", Epochs: ",args["epochs"],", LR: ",args["lr"],", Selection Epochs: ",args["se"])
    print("Optimizer: ",args["optimizer"],", scheduler: ",args["scheduler"],", weight decay: ",args["weight_decay"],", Images: ",num_images,end=" ")
    if args["dataset"]=="cifar100c":
        print(", Corruption: {} %".format(args["corruption_percentage"]))
    else:
        print("\n")
    print("==========================================================\n")

    if args["random_seed"]==0:
        torch.manual_seed(42)
    else:
        torch.manual_seed(random.randint(1,1e6))

    #get the model

    model = get_model(args)
    #get the optimizer and loss function
    optimizer, criterion, scheduler = get_optimizer_criterion(args,model,len(train_loader))
    if args["multi_gpu"]==0:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model,device_ids=[0,1])
        model = model.to(device)

    best_val_acc = 0.0
    test_loss = 0.0
    pbar = tqdm(range(args["epochs"]))
    for epoch in pbar:
        model.train()
        if args["dataset"]=="cifar100c":
            model, train_loss, train_acc = train_model_perturbed(model,train_loader,optimizer,criterion,device,progress_bar=False,scheduler=scheduler,apply_scheduler=apply_scheduler)
        else:
            model, train_loss, train_acc = train_model(model,train_loader,optimizer,criterion,device,progress_bar=args["progress_bar"],scheduler=scheduler,apply_scheduler=apply_scheduler)
        if (epoch+1) % args["test_interval"] == 0 and epoch>args["test_eval_start_epoch"]:
            test_acc, test_loss = test_model(model,test_loader,criterion,device)
           
            if test_acc > best_val_acc:
                best_val_acc = test_acc 
                #checkpoint(model,"saved_models/{}/{}/best_{}_percent.pth".format(args["dataset"],args["model"],int(args["fraction"]*100)))
        pbar.set_postfix_str("train loss: {:.3f}, train acc:{:.3f},test loss:{:.3f}, best test acc: {:.3f}".format(train_loss,train_acc,test_loss,best_val_acc))
        if args["wandb"]==1:
            wandb.log({"train acc":train_acc,"train loss":train_loss,"test acc":best_val_acc,"lr":optimizer.param_groups[0]['lr']})
        if not apply_scheduler:
            scheduler.step(train_loss)
    best_test_acc_list.append(best_val_acc.cpu())
    print("[INFO] Best test accuracy:{:.3f}".format(best_val_acc.cpu()))

print("-------------------------- Completed ------------------")
print("==> Mean Best Accuracy: {:.3f}".format(np.mean(best_test_acc_list)))
print("==> STD Best Accuracy: {:.3f}".format(np.std(best_test_acc_list)))


# recording the details
if args["dataset"]=="cifar100c":
    filename = "progress_tracker/{}_{}_{}_percentage_corruption_results.csv".format(args["dataset"],args["model"],args["corruption_percentage"])
else:
    filename = "progress_tracker/{}_{}_results.csv".format(args["dataset"],args["model"])
fields=['Timestamp','Percentage','Optimizer','Scheduler','batch size','learning rate','weight decay','epochs','selection epochs','iterations','mean accuracy','std accuracy','radius']
if not os.path.exists("progress_tracker"):
    os.mkdir("progress_tracker")

if args["dataset"]=="cifar100c":
    if not os.path.exists("progress_tracker/{}_{}_{}_percentage_corruption_results.csv".format(args["dataset"],args["model"],args["corruption_percentage"])):
        with open(filename,'w') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=fields)
            writer.writeheader()
else:
    if not os.path.exists("progress_tracker/{}_{}_results.csv".format(args["dataset"],args["model"])):
        with open(filename,'w') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=fields)
            writer.writeheader()


result=[{'Timestamp':datetime.now().strftime('%d-%m-%Y %H:%M:%S'),'Percentage':args["fraction"]*100,'Optimizer':args["optimizer"],'Scheduler':args["scheduler"],"batch size":args["batch_size"],'learning rate':"{:.4f}".format(args["lr"]),'weight decay':"{:.4f}".format(args["weight_decay"]),
        'epochs':args["epochs"],'selection epochs':args["se"],'radius':args["radius"],'iterations':args["iterations"],'mean accuracy':"{:.2f}".format(np.mean(best_test_acc_list)),'std accuracy':"{:.3f}".format(np.std(best_test_acc_list))}]

with open(filename,'a+') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writerows(result)



