#imports
import time 
from tqdm import tqdm 
import argparse 
import warnings 
import torch 
import os 
from utils import get_model,get_dataset,checkpoint,resume,train_model,train_model_perturbed

##disable warnings
warnings.warn("userwarning",UserWarning)
#parser
parser = argparse.ArgumentParser(description='PyTorch based Coreset selection')
parser.add_argument("--epochs",type=int,default=10,help='number of epochs for initial selection')
parser.add_argument("--dataset",default='cifar10',type=str,help='chose cifar10,cifar100 or ilsvrc')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--multi_gpu",type=int,default=0)
parser.add_argument("--model",default='resnet18',help='enter resnet18,vgg16,vit_small')
parser.add_argument("--patch",default=4,type=int,help="patch size for ViT model")
parser.add_argument("--dimhead",default=512,type=int)
parser.add_argument("--se",type=int,default=10,help="selection epochs")
parser.add_argument("--num_gpus",type=int,default=1)
parser.add_argument("--corruption_percentage",type=int,default=5)
parser.add_argument("--device",type=str,default="cuda:0",help="specific device id for training")
parser.add_argument("--lr",type=float,default=0.1,help="learning rate for SGD optimizer")
parser.add_argument("--optimizer",default="SGD",help="Enter SGD or Adam")
args = vars(parser.parse_args())

start_time = time.time()
print("======================== Settings ========================")
print("Model:",args["model"]," ,dataset: ",args["dataset"]," ,batch size: ",args["batch_size"])
print("device: ",args["device"]," ,Epochs: ",args["epochs"]," ,LR: ",args["lr"])
print("==========================================================")

args["dataset"]=args["dataset"].lower()
args["model"]=args["model"].lower()

#create the directory to store the neighbors calculation
if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("checkpoint/{}".format(args["dataset"])):
    os.mkdir("checkpoint/{}".format(args["dataset"]))
if not os.path.exists("checkpoint/{}/{}".format(args["dataset"],args["model"])):
    os.mkdir("checkpoint/{}/{}".format(args["dataset"],args["model"]))

#get the dataset
print("\n\n==> Loading the dataset....")
train_loader=get_dataset(args,train=True,gradient_generation=True)

#get the model
print("\n==> Creating the model...")
device = torch.device(args["device"])
model = get_model(args)
if args["multi_gpu"]==0:
    NUM_GPUS=1
else:
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    NUM_GPUS=args["num_gpus"]

if args["optimizer"]=="SGD":
    optimizer = torch.optim.SGD(model.parameters(),lr=args["lr"])
else:
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
criterion = torch.nn.CrossEntropyLoss()
model = model.to(device)

#training loop
print("\n==> Starting the training loop...\n")

for epoch in range(args["epochs"]):
    model.train()
    if args["dataset"]=="cifar100c":
        model, train_loss, train_acc = train_model_perturbed(model,train_loader,optimizer,criterion,device,progress_bar=True)
    else:
        model, train_loss, train_acc = train_model(model,train_loader,optimizer,criterion,device,progress_bar=True)
    print("\n[UPDATE] Epoch: {}, train loss:{:.3f}, train accuracy:{:.3f},time elapsed:{:.3f}\n".format(epoch+1,train_loss,train_acc,time.time()-start_time))
    file=open("model_training_loss_accuracy.txt","a+")
    string_to_write=str(epoch)+","+str(train_loss)+","+str(train_acc)
    file.writelines(string_to_write)
    file.writelines("\n")
    file.close()    
    if args["dataset"]=="cifar100c":
        checkpoint(model,"checkpoint/{}/{}_{}_percent/saved_model_{}.pth".format(args["dataset"],args["model"],args["corruption_percentage"],epoch),args["multi_gpu"])
        print("Saved to checkpoint/{}/{}_{}_percent/saved_model_{}.pth".format(args["dataset"],args["model"],args["corruption_percentage"],epoch))
    else:
        checkpoint(model,"checkpoint/{}/{}/saved_model_{}.pth".format(args["dataset"],args["model"],epoch),args["multi_gpu"])
        print("Saved to checkpoint/{}/{}/saved_model_{}.pth".format(args["dataset"],args["model"],epoch))





end_time = time.time()
print("[INFO] Training time for single epoch:{:.3f} seconds.".format((end_time-start_time)/args["epochs"]))

