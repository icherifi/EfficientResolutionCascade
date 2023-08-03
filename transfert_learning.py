
import torch
import torchvision.models as models
import time
import torch.optim as optim
from data_loader import ImageNetDataLoader
from train_utils import train
from scores import validate


class TransferLearningResNet :
    
    def __init__(self, model, loader, epochs=1, path_to_model = None, path_to_save = None):
        self.model = model
        self.loader = loader
        self.path_to_model = path_to_model
        if path_to_model == None :
            net = self.get_net().to('cuda')
            pretrain_net = self.train(net ,epochs)
            path_models_folder = path_to_save
            torch.save(pretrain_net, path_models_folder + '/' + model + '_resolution_'+ str(self.loader.resolution) + '.pth')
        else :
            net = self.load_net().to('cuda')
        
    def load_net(self):
        print("Loading model from file")
        net = torch.load(self.path_to_model)
        return net
        
    def get_net(self):
        if self.model == "resnet18" :
            resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        if self.model == "resnet34" :
            resnet = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        if self.model == "resnet50" :
            resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        if self.model == "resnet101" :
            resnet = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
        if self.model == "resnet152" :
            resnet = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V1')
            
        total_layers = sum(1 for _ in resnet.named_parameters())
        print(f"Total number of layers in the model: {total_layers}")
        print(f"Model beeing trained: {self.model}")
                        
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(resnet.fc.weight)
        return resnet

    def train(self, net, epochs=10):
        device='cuda'

        #criterion = torch.nn.CrossEntropyLoss()
        #optimizer = optim.SGD(net.fc.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        print(f'Training for {epochs} epochs on {device}')
        for epoch in range(epochs):
            train(self.loader, net, criterion, optimizer, epoch)
            val_loader = self.loader.data_batch["val"]
            acc1 = validate(val_loader, net, criterion)
            scheduler.step()
        return net
