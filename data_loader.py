import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time

class ImageNetDataLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir=None): #can add transformer as parameter
        self.resolution = 256
        self.root_dir = root_dir
        self.batch_size = 32
        self.data = {
                    "val_id":[],
                    "train": [],
                    "val": [],
                    }
        self.data_batch = {
                    "val_id":[],
                    "train": [],
                    "val": [],
                    }
        self.image_id = []
            
    def shape(self):
        traindir = os.path.join(self.root_dir, 'train')
        valdir = os.path.join(self.root_dir, 'val')
        train_dataset = datasets.ImageFolder(root=traindir)
        val_dataset = datasets.ImageFolder(root=valdir)
        return len(train_dataset), len(val_dataset)
        
    def get_data(self, resolution = 256):
        self.resolution = resolution
        print("Getting data ........")
        print("Resolution : ", self.resolution)
        traindir = os.path.join(self.root_dir, 'train')
        valdir = os.path.join(self.root_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(resolution),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(degrees=(0, 10), translate=(0.1, 0.1), shear=(0, 10)), 79.745; 79.924 ;80.637; 81.783; 80.153 
                transforms.RandomAffine(degrees=(0, 10), translate=(0.1, 0.1), shear=(0, 10)),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(resolution),
                transforms.ToTensor(),
                normalize,
            ]))
        
        print("Number of training data :", len(train_dataset))
        print("Number of validating data :", len(val_dataset))
        self.data["val_id"] = [i for i in range(len(val_dataset))]
        self.data["train"] = train_dataset
        self.data["val"] = val_dataset
        return self
        
    def load_data(self, fromFile = True):
        print("Loading data ........")
        
        if self.data["train"] != []:
            train_loader = torch.utils.data.DataLoader(
                self.data["train"], batch_size=self.batch_size, shuffle=True)
            self.data_batch["train"] = train_loader
            
        val_loader = torch.utils.data.DataLoader(
            self.data["val"], batch_size=self.batch_size, shuffle=False)
        self.data_batch["val"] = val_loader
        
        self.data_batch["val_id"] = [self.data["val_id"][i:i+self.batch_size] for i in range(0, len(self.data["val_id"]), self.batch_size)]

        return self

    def select_data(self, list_id):
        data_selected = ImageNetDataLoader()
        data_selected.data["val_id"] = list_id
        data_selected.data["val"] = torch.utils.data.Subset(self.data["val"], list_id)
        return data_selected.load_data(fromFile=False)
    

# Loading dataset
#data_dir = "C:/Users/asily/cnn_efficience/cascade_cnn_imagenette/imagenette2-320"
#dataset = ImageNetDataLoader(data_dir, (32,32))
#d = dataset.select_data([10,11,43,54,76])
#d.select_data([43,76])