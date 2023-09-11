
import torch
import torchvision.models as models
import time
import torch.optim as optim
from data_loader import ImageNetDataLoader
from train_utils import train
from scores import validate
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR


class TransferLearningResNet :
    
    def __init__(self, model, loader, epochs=1, path_to_model = None, path_to_save = None, lr = 0.01):
        self.model = model
        self.loader = loader
        self.path_to_model = path_to_model
        self.lr = lr
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
            #resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            
            resnet = models.resnet18(weights=None)
        if self.model == "resnet34" :
            resnet = models.resnet34(weights=None)
        if self.model == "resnet50" :
            resnet = models.resnet50(weights=None)
        if self.model == "resnet101" :
            resnet = models.resnet101(weights=None)
        if self.model == "resnet152" :
            resnet = models.resnet152(weights=None)
            
        total_layers = sum(1 for _ in resnet.named_parameters())
        print(f"Total number of layers in the model: {total_layers}")
        print(f"Model beeing trained: {self.model}")
                        
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
        #resnet.fc = torch.nn.Linear(512, 10)
        torch.nn.init.xavier_uniform_(resnet.fc.weight)
        return resnet

    def train(self, net, epochs=10):
        device='cuda'

        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(net.parameters(), lr=self.lr)

        # Decay LR by a factor of 0.1 every 30 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1) #90 epochs is good
                
        print(f'Training for {epochs} epochs on {device}')
        best_model = deepcopy(net)
        best_acc = None
        for epoch in range(epochs):
            
            train(self.loader, net, criterion, optimizer, epoch)
            val_loader = self.loader.data_batch["val"]
            acc1 = validate(val_loader, net, criterion)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            # Afficher le learning rate à chaque époque
            print(f"Epoch [{epoch}/{epochs}], Learning Rate: {current_lr}")
                    
            if best_acc is None or best_acc < acc1:
                best_acc = acc1
                best_model = deepcopy(net)
                
        print("Best accurancy : {:.2f}%".format(best_acc))
        return best_model
    
    
