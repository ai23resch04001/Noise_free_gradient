from models import resnet
from models.vit_small import ViT
from models.vgg16 import VGG16
from torch import nn 
from torchvision import transforms , datasets
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm 
import numpy as np 
import pickle 
import random 
import torch 
import math 
from datasets.dataset import CIFAR100Corrupt
import os 
import json 
from PIL import Image 
#function to generate ranking 
def get_indices(indices,radius):
    indices_new=[]
    a=[np.argwhere(x<radius).reshape(-1) for x in indices]
    indices_new=[indices[i][a[i]] for i in range(len(a))]
    return indices_new

def get_ranking_multiprocessing(indices):
    length_list_i=[]
    for i in range(len(indices)):
        length_list_i.append(len(indices[i]))
    ranking_list=np.argsort(length_list_i)[::-1]
    return ranking_list


#class definition
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "Datasets/ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":

                syn_id = entry
                syn_id_t=syn_id.split("_")[0]
                target = self.syn_to_class[syn_id_t]
                sample_path = os.path.join(samples_dir, syn_id)
                #for sample in os.listdir(syn_folder):
                    #sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(target)
            elif split == "val":
                # entry_t1=entry.split("_")
                # entry_t2="_".join(entry_t1[:-1])+".JPEG"
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]



def get_coreset(program_args,train_dataset,num_classes):
    print("==> Preparing Ranking ...")
    print("\n==> Using folder : gradients_folder/{}/{}".format(program_args["dataset"],program_args["model"]))
    ranking_dict={}
    #print(num_classes,program_args["se"])
    for i in tqdm(range(num_classes)):
        for j in range(program_args["se"]):
            with open('gradients_folder/{}/{}/neighbors_{}_{}.pkl'.format(program_args["dataset"],program_args["model"],i,j),'rb') as f:
                indices=pickle.load(f)
                indices = get_indices(indices,program_args["radius"])
                internal_ranking=get_ranking_multiprocessing(indices)
                if j==0:
                    internal_ranking_list=np.zeros(len(internal_ranking))
                    for k in range(len(internal_ranking)):
                        internal_ranking_list[internal_ranking[k]] = k
                else:
                    for k in range(len(internal_ranking)):
                        internal_ranking_list[internal_ranking[k]] += k

        ranking_dict[i]=np.argsort(internal_ranking_list)


    #prepare coreset
    f=open('mapping_data_{}.pkl'.format(program_args["dataset"]),'rb')
    mapping_dict=pickle.load(f)
    top_N_indices_list=[]


    for classes in ranking_dict.keys():
        required_num=round(program_args["fraction"]*len(ranking_dict[classes]))
        # print(classes,required_num,len(ranking_dict[classes]))
        if required_num==0:
            required_num=1
        


        for i in range(required_num):
            top_N_indices_list.append(mapping_dict[classes][ranking_dict[classes][i]])

    random.shuffle(top_N_indices_list)
    coreset_dataset=torch.utils.data.Subset(train_dataset,top_N_indices_list)
    print("Total number of images in coreset: ",len(coreset_dataset))
    print("==> Selecting coreset ")
    trainloader=torch.utils.data.DataLoader(
            coreset_dataset,
            batch_size=program_args["batch_size"],
            shuffle=True,
            pin_memory = True,
            drop_last = True,
            num_workers=4*program_args["num_gpus"])
            #num_workers=16)
    return trainloader,len(coreset_dataset)



def get_dataset(program_args,train=True,gradient_generation=False,gradient_calculation=False,shuffle=True):
    
    if program_args["dataset"]=="cifar10":
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if gradient_generation:  # No test data is required
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR10('./data',download=True,train=True,transform=train_transform)
            train_loader = DataLoader(
                        train_dataset,
                        batch_size=program_args["batch_size"], # may need to reduce this depending on your GPU 
                        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                        shuffle=shuffle,
                        drop_last=False
                    )
            return train_loader 
        elif gradient_calculation:
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR10('./data',download=True,train=True,transform=train_transform)
            return train_dataset
        else:
            if train:
                train_transform=transforms.Compose(
                    [
                    transforms.RandAugment(),
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
                )
                train_dataset=datasets.CIFAR10('./data',download=True,train=True,transform=train_transform)
                return get_coreset(program_args,train_dataset,num_classes)
            else:
                test_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                test_dataset=datasets.CIFAR10('./data',download=True,train=False,transform=test_transform)
                test_loader = DataLoader(
                            test_dataset,
                            batch_size=128, # may need to reduce this depending on your GPU 
                            num_workers=16, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False
                        )
                return test_loader 

    elif program_args["dataset"]=="cifar10n":
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if gradient_generation:  # No test data is required
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR10('./data',download=True,train=True,transform=train_transform)
            train_dataset.targets=np.load('data/cifar-10-noisy/cifar10.npy')
            train_loader = DataLoader(
                        train_dataset,
                        batch_size=program_args["batch_size"], # may need to reduce this depending on your GPU 
                        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                        shuffle=shuffle,
                        drop_last=False
                    )
            return train_loader 
        elif gradient_calculation:
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR10('./data',download=True,train=True,transform=train_transform)
            train_dataset.targets=np.load('data/cifar-10-noisy/cifar10.npy')
            return train_dataset
        else:
            if train:
                train_transform=transforms.Compose(
                    [
                    transforms.RandAugment(),
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
                )
                train_dataset=datasets.CIFAR10('./data',download=True,train=True,transform=train_transform)
                train_dataset.targets=np.load('data/cifar-10-noisy/cifar10.npy')
                return get_coreset(program_args,train_dataset,num_classes)
            else:
                test_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                test_dataset=datasets.CIFAR10('./data',download=True,train=False,transform=test_transform)
                test_loader = DataLoader(
                            test_dataset,
                            batch_size=128, # may need to reduce this depending on your GPU 
                            num_workers=16, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False
                        )
                return test_loader 
            
        
    elif program_args["dataset"]=="cifar100":
        num_classes = 100
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        if gradient_generation:  # No test data is required
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
            train_loader = DataLoader(
                        train_dataset,
                        batch_size=program_args["batch_size"], # may need to reduce this depending on your GPU 
                        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                        shuffle=shuffle,
                        drop_last=False
                    )
            return train_loader 
        elif gradient_calculation:
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
            return train_dataset
        else:
            if train:
                train_transform=transforms.Compose(
                    [
                    transforms.RandAugment(),
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
                )
                train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
                return get_coreset(program_args,train_dataset,num_classes)
            else:
                test_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                test_dataset=datasets.CIFAR100('./data',download=True,train=False,transform=test_transform)
                test_loader = DataLoader(
                            test_dataset,
                            batch_size=128, # may need to reduce this depending on your GPU 
                            num_workers=16, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False
                        )
                return test_loader 

    elif program_args["dataset"]=="cifar100n":
        num_classes = 100
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        if gradient_generation:  # No test data is required
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
            train_dataset.targets=np.load('data/cifar-100-noisy/cifar.npy')
            train_loader = DataLoader(
                        train_dataset,
                        batch_size=program_args["batch_size"], # may need to reduce this depending on your GPU 
                        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                        shuffle=shuffle,
                        drop_last=False
                    )
            return train_loader 
        elif gradient_calculation:
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
            train_dataset.targets=np.load('data/cifar-100-noisy/cifar.npy')
            return train_dataset
        else:
            if train:
                train_transform=transforms.Compose(
                    [
                    transforms.RandAugment(),
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
                )
                train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
                train_dataset.targets=np.load('data/cifar-100-noisy/cifar.npy')
                return get_coreset(program_args,train_dataset,num_classes)
            else:
                test_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                test_dataset=datasets.CIFAR100('./data',download=True,train=False,transform=test_transform)
                test_loader = DataLoader(
                            test_dataset,
                            batch_size=128, # may need to reduce this depending on your GPU 
                            num_workers=16, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False
                        )
                return test_loader 


    elif program_args["dataset"]=="cifar100c":
        num_classes = 100
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        if gradient_generation:  # No test data is required
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            # train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
            train_dataset=CIFAR100Corrupt(root='data',transform=train_transform)
            train_loader = DataLoader(
                        train_dataset,
                        batch_size=program_args["batch_size"], # may need to reduce this depending on your GPU 
                        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                        shuffle=shuffle,
                        drop_last=False
                    )
            return train_loader 
        elif gradient_calculation:
            train_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            # train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
            train_dataset=CIFAR100Corrupt(root='data',transform=train_transform)
            return train_dataset
        else:
            if train:
                train_transform=transforms.Compose(
                    [
                    transforms.RandAugment(),
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
                )
                # train_dataset=datasets.CIFAR100('./data',download=True,train=True,transform=train_transform)
                train_dataset=CIFAR100Corrupt(root='data',transform=train_transform)
                return get_coreset(program_args,train_dataset,num_classes)
            else:
                test_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                test_dataset=datasets.CIFAR100('./data',download=True,train=False,transform=test_transform)
                test_loader = DataLoader(
                            test_dataset,
                            batch_size=128, # may need to reduce this depending on your GPU 
                            num_workers=16, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False
                        )
                return test_loader 

    elif program_args["dataset"]=="ilsvrc":
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if gradient_generation:  # No test data is required
            train_transform = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.ImageFolder(root='data/ILSVRC/Data/CLS-LOC/train', transform=train_transform)
            train_loader = DataLoader(
                        train_dataset,
                        batch_size=program_args["batch_size"], # may need to reduce this depending on your GPU 
                        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                        shuffle=shuffle,
                        drop_last=False
                    )
            return train_loader 
        elif gradient_calculation:
            train_transform = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ]
                    )
            train_dataset=datasets.ImageFolder(root='data/ILSVRC/Data/CLS-LOC/train', transform=train_transform)
            return train_dataset
        else:
            if train:
                train_transform=transforms.Compose(
                    [
                    transforms.RandAugment(),
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
                )
                train_dataset=datasets.ImageFolder(root='data/ILSVRC/Data/CLS-LOC/train', transform=train_transform)
                return get_coreset(program_args,train_dataset,num_classes)
            else:
                test_transform = transforms.Compose(
                            [
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                test_dataset=ImageNetKaggle("data", "val", test_transform)
                test_loader = DataLoader(
                            test_dataset,
                            batch_size=128, # may need to reduce this depending on your GPU 
                            num_workers=16, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False
                        )
                return test_loader

    else:
        raise NotImplementedError("Dataset {} is not a valid dataset".format(program_args["dataset"]))


#functions
#model creationi
def checkpoint(model,filename,multi_gpu=False):
    if multi_gpu:
        torch.save(model.module.state_dict(),filename)
    else:
        torch.save(model.state_dict(),filename)
def resume(model,filename,multi_gpu=False):
    if multi_gpu:
        model.module.load_state_dict(torch.load(filename))
    else:
        model.load_state_dict(torch.load(filename,map_location=torch.device('cuda:0')))

        
def get_model(program_args):
    dimhead = 512
    depth = 6
    num_heads = 8
    if program_args["dataset"]=="cifar10":
        num_classes = 10
        im_size = 32
        patch_size=4
    elif program_args["dataset"]=="cifar100" or program_args["dataset"]=="cifar100c" or program_args["dataset"]=="cifar100n":
        num_classes = 100
        im_size = 32
        patch_size = 4


    else:
        num_classes = 1000
        im_size = 224
        patch_size = 16
        dimhead = 768
        depth = 12 
        num_heads = 12

    if program_args["model"]=="vit_small":
        model = ViT(
            image_size = im_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dimhead,
            depth = depth ,
            heads = num_heads,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif program_args["model"]=="resnet18":
        model = resnet.ResNet18(channel=3,num_classes=num_classes,im_size=[im_size,im_size])
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,a=math.sqrt(3))

            if isinstance(m,nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,0.975)
                torch.nn.init.constant_(m.bias,0.125)


        if program_args["dataset"]=='ilsvrc':
            n = model.fc.in_features
            model.fc = nn.Linear(n,128)
            model = nn.Sequential(model,nn.ReLU(),nn.Linear(128,num_classes))

    elif program_args["model"]=="resnet50":
        model = resnet.ResNet50(channel=3,num_classes=num_classes,im_size=[im_size,im_size])


        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,a=math.sqrt(3))

            if isinstance(m,nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,0.975)
                torch.nn.init.constant_(m.bias,0.125)


        if program_args["dataset"]=='ilsvrc':
            n = model.fc.in_features
            model.fc = nn.Linear(n,128)
            model = nn.Sequential(model,nn.ReLU(),nn.Linear(128,num_classes))

    elif program_args["model"]=="resnet18extractor":
        model = resnet.ResNet18Extractor(channel=3,num_classes=num_classes,im_size=[im_size,im_size])
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,a=math.sqrt(3))

            if isinstance(m,nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,0.975)
                torch.nn.init.constant_(m.bias,0.125)




    elif program_args["model"]=="resnet50":
        model = resnet.ResNet50(channel=3,num_classes=num_classes,im_size=[im_size,im_size])
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,a=math.sqrt(3))

            if isinstance(m,nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,0.975)
                torch.nn.init.constant_(m.bias,0.125)


        if program_args["dataset"]=='ilsvrc':
            n = model.fc.in_features
            model.fc = nn.Linear(n,128)
            model = nn.Sequential(model,nn.ReLU(),nn.Linear(128,num_classes))

    elif program_args["model"]=='vgg':
        model = VGG16(num_classes=num_classes)
        if program_args["dataset"]=="ilsvrc":
            n = model.fc.in_features 
            model.fc = nn.Linear(n,128)
            model = nn.Sequential(model,nn.ReLU(),nn.Linear(128,num_classes))
    else:
        raise NotImplementedError("{} is not a valid model name.".format(program_args["model"]))
    return model 


def train_model_perturbed(model,train_loader,optimizer,criterion,device,progress_bar=False,scheduler=None,apply_scheduler=False):
    total_loss = 0.0
    train_acc = 0.0
    total_images = 0
    if progress_bar:
        pbar = tqdm(range(len(train_loader)))
        for _,images,labels in train_loader:
            images , labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            y_pred = torch.argmax(output,axis=1)
            train_acc += torch.sum(y_pred==labels)
            total_images += len(images)
            pbar.update()
            if scheduler is not None and apply_scheduler==True:
                scheduler.step()
    else:
        for _,images,labels in train_loader:
            images , labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            y_pred = torch.argmax(output,axis=1)
            train_acc += torch.sum(y_pred==labels)
            total_images += len(images)
            if scheduler is not None and apply_scheduler==True:
                scheduler.step()
    train_acc = train_acc.cpu().detach().numpy()*100/total_images
    return model,total_loss,train_acc

def train_model(model,train_loader,optimizer,criterion,device,progress_bar=False,scheduler=None,apply_scheduler=False):
    total_loss = 0.0
    train_acc = 0.0
    total_images = 0
    if progress_bar:
        pbar = tqdm(range(len(train_loader)))
        for images,labels in train_loader:
            images , labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            y_pred = torch.argmax(output,axis=1)
            train_acc += torch.sum(y_pred==labels)
            total_images += len(images)
            pbar.update()
            if scheduler is not None and apply_scheduler==True:
                scheduler.step()
    else:
        for images,labels in train_loader:
            images , labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            y_pred = torch.argmax(output,axis=1)
            train_acc += torch.sum(y_pred==labels)
            total_images += len(images)
            if scheduler is not None and apply_scheduler==True:
                scheduler.step()
    train_acc = train_acc.cpu().detach().numpy()*100/total_images
    return model,total_loss,train_acc

def test_model(model,test_loader,criterion,device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    total = 0
    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            y_pred = torch.argmax(outputs,axis=1)
            test_acc += torch.sum(y_pred == targets)
            total += len(inputs)
    test_acc = 100.0*test_acc/total 
    test_loss = test_loss / total 
    return test_acc ,test_loss 

def test_model_class_wise(model,test_loader,criterion,device,num_classes=100):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    total = 0
    class_wise_accuracy=np.zeros(num_classes)
    class_wise_total=np.zeros(num_classes)
    pbar = tqdm(range(len(test_loader)))
    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            y_pred = torch.argmax(outputs,axis=1)
            for i in range(len(y_pred)):
                if y_pred[i]==targets[i]:
                    class_wise_accuracy[targets[i]]+=1
                class_wise_total[targets[i]]+=1
            pbar.update(1)

    for i in range(num_classes):
        class_wise_accuracy[i] /= class_wise_total[i]
    return class_wise_accuracy


def get_optimizer_criterion(args,model,train_loader_length=1):
    if args["optimizer"]=="SGD":
        optimizer=torch.optim.SGD(model.parameters(),lr=args["lr"],momentum=0.9,weight_decay=args["weight_decay"],nesterov=True)
    else:
        optimizer=torch.optim.Adam(model.parameters(), lr=args["lr"],weight_decay=args["weight_decay"])

    if args["scheduler"]=="reduce_lr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2,factor=0.9)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"]*train_loader_length,eta_min=args["min_lr"])

    criterion = torch.nn.CrossEntropyLoss()

    return optimizer,criterion,scheduler
