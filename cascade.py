from data_loader import ImageNetDataLoader
import torchvision.models as models
import torch
from scores import validate
import numpy as np
import itertools
import time
import json

def find_prediction(list, x, res, thresh):
    for element in list:
        if element.get_id() == x and element.get_resolution() == res and torch.any(element.get_prediction() > thresh):
            return element
    return None

class PredictionCC:
    def __init__(self, id = None, prediction = None, classe = None, ground_truth = None ,resolution = None):
        self.id =  id
        self.prediction = prediction
        self.classe = classe
        self.resolution = resolution
        self.ground_truth = ground_truth
    def get_id(self):
        return self.id
    def get_prediction(self):
        return self.prediction
    def get_classe(self):
        return self.classe
    def get_ground_truth(self):
        return self.ground_truth
    def get_resolution(self):
        return self.resolution


class Cascade:
    def __init__(self, data_dir, thresolds, models, resolutions, loaders = None):
        self.data_dir = data_dir
        self.thresholds = thresolds
        self.models = models
        self.resolutions = resolutions
        self.predictions = []
        if loaders == None:
            self.loaders = self.load_data()
        else:
            self.loaders = loaders
        self.stats = self.Stats(self.predictions, self.resolutions, self.models, self.data_dir)
    
    def load_data(self):
        loaders = []
        for res in self.resolutions :
            dataset = ImageNetDataLoader(self.data_dir).get_data(resolution=res)
            loader = dataset.load_data()
            loaders.append(loader)
        return loaders
    
    def predict(self, loader, model):
        print("Predicting .........")
        device = "cuda"
        with torch.no_grad():
            pred = []
            id = []
            gtruth = []
            for batch_id , (images, target) in zip(loader.data_batch['val_id'], loader.data_batch['val']):
                images = images.to(device, non_blocking=True)
                softmax =  torch.nn.Softmax(dim=1)
                outputs = model(images)
                pred.append(softmax(outputs))
                gtruth.append(target)
                id.append(batch_id)
            return zip(id, pred, gtruth)
    
    def fill_predictionCC(self, loader, model, resolution, threshold=0):
        outputs = self.predict(loader=loader, model=model)                  
        for batch_id, batch_preds, batch_gtruth in outputs:
            for id, pred, gtruth in zip(batch_id, batch_preds, batch_gtruth):
                if torch.any(pred > threshold) :
                    if self.list_id_unpredicted != None :
                        self.list_id_unpredicted.remove(id)
                    predcc = PredictionCC(
                        id = id, 
                        prediction = pred, 
                        classe = torch.argmax(pred), 
                        ground_truth = gtruth,
                        resolution = resolution
                        )
                    self.predictions.append(predcc)
        
    def global_predict(self):
        load = ImageNetDataLoader(root_dir = self.data_dir)
        self.list_id_unpredicted = [i for i in range(load.shape()[1])]

        for threshold, model, resolution, loader in zip(self.thresholds, self.models[:-1], self.resolutions[:-1], self.loaders[:-1]):
            loader = loader.select_data(list_id=self.list_id_unpredicted)

            self.fill_predictionCC(
                loader= loader, 
                model = model, 
                resolution = resolution, 
                threshold=threshold, 
                )
            
        model, resolution = self.models[-1], self.resolutions[-1]
        loader = self.loaders[-1]
        loader = loader.select_data(list_id=self.list_id_unpredicted)
        self.fill_predictionCC(
                loader=loader, 
                model= model, 
                resolution = resolution, 
                )
                    
    def found_threshold(self, init_thresh = None, step = 10, target_acc = None, r_dir = "."):
        if init_thresh == None:
            init_thresh = [0 for i in range(dim)]
        self.list_id_unpredicted = None
        
        #for model, loader, resolution in zip(self.models, self.loaders, self.resolutions):
            #self.fill_predictionCC(
                #loader=loader, 
                #model = model, 
                #resolution = resolution, 
                #) #prediction is filled with all
            
        dim = len(self.models)
        r_time = []
        r_acc = []
        r_num_images = []    
        thresh = np.linspace(init_thresh, 1, step)
        threshs = list(itertools.product(thresh, repeat=dim-1))
        for thresh in threshs :
            
            cc = Cascade(self.data_dir, thresh, self.models, self.resolutions, self.loaders)
            
            start_time = time.time()  
            cc.global_predict()
            end_time = time.time()
            
            t = end_time - start_time
            r_time.append(t)
            
            acc = cc.stats.eval_acc()
            r_acc.append(acc)
            
            num_images = cc.stats.eval_resolutions()
            r_num_images.append(num_images)
        
        with open(r_dir + '/accurancy.json', 'w') as file:
            json.dump(r_acc, file)

        with open(r_dir + '/cascadeProcessingTime.json', 'w') as file:
            json.dump(r_time, file)
            
        np.savetxt(r_dir + '/nbImagesProcessed.csv', r_num_images, delimiter=',', header='model0, model1', comments='')
                
    class Stats:
        def __init__(self, predictions, resolutions, models, data_dir):
            self.predictions = predictions
            self.resolutions = resolutions
            self.models = models
            self.data_dir = data_dir 
        def eval_resolutions(self):
            num_per_resolution = []
            for resolutions in self.resolutions :
                count = 0
                for prediction in self.predictions:
                    if prediction.get_resolution() == resolutions:
                        count += 1 
                print("Number of elt predicted at resolution ", resolutions, " : ", count)
                num_per_resolution.append(count)
            return num_per_resolution
        def eval_acc(self):
            acc = 0
            for prediction in self.predictions:
                acc += int(prediction.get_ground_truth() == prediction.get_classe())
            acc = acc/len(self.predictions)
            acc = acc * 100
            print("Accuracy: {:.2f}%".format(acc))
            return acc
        def eval_models(self):
            i=0
            for model in self.models :
                i+=1
                loader = ImageNetDataLoader(root_dir = self.data_dir, resolution=32).load_data() #Change to load only val data
                criterion = torch.nn.CrossEntropyLoss().cuda()
                val_loader = loader.data_batch["val"]
                acc = validate(val_loader, model, criterion, mute=True)
                print("Model{} Accuracy: {:.2f}%".format(i, acc))
                    
batch_size = 32   
data_dir = "C:/Users/asily/cnn_efficience/cascade_cnn_imagenette/imagenette2-320"
model_path = 'C:/Users/asily/cnn_efficience/cascade_cnn/PyTorch/EfficientCascade/Models/resnet18_resolution_32.pth'
model = torch.load(model_path)
cc = Cascade(data_dir=data_dir, thresolds = None, models = [model, model, model], resolutions = [64, 48, 32])
#cc.global_predict()
#cc.stats.eval_resolutions()
#cc.stats.eval_models()
#cc.stats.eval_acc()

cc.found_threshold(init_thresh = 0.5, step = 3, target_acc = None, r_dir = "C:/Users/asily/cnn_efficience/cascade_cnn/PyTorch/EfficientCascade/Results")