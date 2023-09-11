import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from torchvision.utils import save_image
from torch.utils.data import random_split
import numpy as np
import copy

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class ImageNetDataLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, batch_size = 32): #can add transformer as parameter
        self.resolution = 256
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.data = {
                    "train": [],
                    "val": [],
                    "test" : [],
                    }
        self.data_batch = {
                    "train": [],
                    "val": [],
                    "test" : [],
                    }
        self.image_id = []
        self.full_targets = []
        
    def get_data(self, resolution = 256, pre_resize = False):
        self.resolution = resolution
        print("Resolution : ", self.resolution)
        traindir = os.path.join(self.root_dir, 'train')
        valdir = os.path.join(self.root_dir, 'val')
        
        if not pre_resize : 
            print("Data augmentation...")
            
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.Resize(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=(-30, 30), translate=(0.3, 0.3), shear=(0, 30)),
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

            
        if pre_resize :
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=(-30, 30), translate=(0.3, 0.3), shear=(0, 30)),
                    transforms.ToTensor(),
                    normalize,
                ]))
            
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
 
        seed = 42
        np.random.seed(seed)
        val_size = int(len(val_dataset)*0.7)
        val_indices = list(range(len(val_dataset)))
        np.random.shuffle(val_indices)
            
        val_indices, test_indices = val_indices[:val_size], val_indices[val_size:]

        val_dataset_ = torch.utils.data.Subset(val_dataset, val_indices)
        val_dataset.indices = list(range(len(val_dataset_)))
        
        test_dataset = torch.utils.data.Subset(val_dataset, test_indices)
        test_dataset.indices = list(range(len(test_dataset)))
        
        print("Number of training data :", len(train_dataset))
        print("Number of validating data :", len(val_dataset_))
        print("Number of test data :", len(test_dataset))
        self.data["train"] = train_dataset
        self.data["test"] = test_dataset
        self.data["val"] = val_dataset_
        return self
    
        
    def load_data(self, fromFile = True):
            
        if self.data["train"] != []:
            train_loader = torch.utils.data.DataLoader(
                self.data["train"], batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.data_batch["train"] = train_loader
            
        val_loader = torch.utils.data.DataLoader(
            self.data["val"], batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.data_batch["val"] = val_loader
        
        test_loader = torch.utils.data.DataLoader(
            self.data["test"], batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.data_batch["test"] = test_loader
                  
        return self
    
    def apply_transformations_and_save(self, save_dir, resolution):
        # Apply transformations and save transformed images
        
        self.resolution = resolution
        print("Resolution : ", self.resolution)
        traindir = os.path.join(self.root_dir, 'train')
        valdir = os.path.join(self.root_dir, 'val')
        val_dataset = datasets.ImageFolder(
                valdir,
                transforms.ToTensor())
        train_dataset = datasets.ImageFolder(
                traindir,
                transforms.ToTensor())
        
        for idx, (image, target) in enumerate(train_dataset):
            
            transformed_image = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(resolution),
            ])(image)
            class_dir = os.path.join(save_dir, "train" , str(target))  # Create a directory for the class
            os.makedirs(class_dir, exist_ok=True)
            transformed_image_path = os.path.join(class_dir, f"image_{idx}.jpg")
            save_image(transformed_image, transformed_image_path)
            print(f"Transformed image saved at: {transformed_image_path}")
        
        for idx, (image, target) in enumerate(val_dataset):
            transformed_image = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(resolution),
            ])(image)
            
            class_dir = os.path.join(save_dir, "val" , str(target))  # Create a directory for the class
            os.makedirs(class_dir, exist_ok=True)
            transformed_image_path = os.path.join(class_dir, f"image_{idx}.jpg")
            save_image(transformed_image, transformed_image_path)
            print(f"Transformed image saved at: {transformed_image_path}")

    def select_data(self, list_id, subset_name = "val"):
        data_selected = ImageNetDataLoader()
        data_selected.data[subset_name] = torch.utils.data.Subset(self.data[subset_name], list_id)
        data_selected.data[subset_name].indices = list_id
        return data_selected.load_data(fromFile=False)
    
#Loader = ImageNetDataLoader(root_dir='C:\\Users\\asily\\cnn_efficience\\cascade_cnn_imagenette\\imagenette2-320', batch_size=32)
#Loader.apply_transformations_and_save(save_dir='imagenette_resize\\96', resolution=96)
