import argparse 
from tqdm import tqdm 
from utils import get_dataset
import warnings 
import pickle
##disable warnings
warnings.warn("userwarning",UserWarning)
parser = argparse.ArgumentParser(description='PyTorch based Coreset selection')
parser.add_argument("--dataset",default='cifar100',type=str,help='chose cifar10,cifar100 or ilsvrc')
parser.add_argument("--batch_size",type=int,default=128)
args = vars(parser.parse_args())

args["dataset"]=args["dataset"].lower()
train_loader = get_dataset(args,train=True,gradient_generation=True,shuffle=False,gradient_calculation=False)

if args["dataset"]=='cifar10':
    NUM_CLASSES = 10
elif args["dataset"]=='cifar100':
    NUM_CLASSES = 100
elif args["dataset"]=="tiny":
    NUM_CLASSES = 200
else:
    NUM_CLASSES = 1000

count_dict={}
for classes in range(NUM_CLASSES):
    count_dict[classes]=[]

pbar=tqdm(range(len(train_loader)),position=0,leave=True)
counter=0
for images,labels in train_loader:
    classes=labels.cpu().detach().numpy()
    for i in range(len(classes)):
        count_dict[classes[i]].append(counter*128+i)
    counter +=1
    pbar.update()

f=open("mapping_data_{}.pkl".format(args["dataset"]),"wb")
pickle.dump(count_dict,f)
