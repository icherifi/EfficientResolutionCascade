from data_loader import ImageNetDataLoader
import torchvision.models as models
import torch
from scores import validate
import numpy as np
import itertools
import time
import json


seed = 42
np.random.seed(seed)

def ResNetflop(image_props): #Props beteween 0 and 1
    ip = image_props
    FLOP0 = 11.69*(32/224)**2 #Million flops for resnet18 on 32x32 images
    FLOP1 = 21.80*(96/224)**2 #Million flops for resnet50 on 96x96 images
    FLOP2 = 25.56 #Million flops for resnet152 on 224x224 images
    
    ccFLOP = FLOP2*(1-ip[0]-ip[1]) + FLOP1*(1-ip[0]) + FLOP0*1
    return ccFLOP

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
    def __init__(self, data_dir, thresolds, models, resolutions, targets = None, loaders = None, pre_pred = None, subset_name = "val"):
        self.data_dir = data_dir
        self.thresholds = thresolds
        self.models = models
        self.resolutions = resolutions
        self.predictions = []
        self.models_predictions = pre_pred
        self.global_targets = targets
        self.subset_name = subset_name

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
    
    def pre_pred(self):
        device = "cuda"
        preds = []
        with torch.no_grad() :
            for model, loader in zip(self.models, self.loaders):
                pred = []
                for images, _ in loader.data_batch[self.subset_name]:
                    images = images.to(device, non_blocking=True)
                    softmax =  torch.nn.Softmax(dim=1)
                    outputs = softmax(model(images))
                    pred.append(outputs)
                preds.append(pred)
            print("End of pre-prediction")
            return preds
                
    
    def predict(self, loader, model):
        print("Predicting .........")
        device = "cuda"
        with torch.no_grad():
            pred = []
            id = []
            gtruth = []
            for batch_id , (images, target) in zip(loader.data_batch[self.subset_name].indices, loader.data_batch[self.subset_name]):
                images = images.to(device, non_blocking=True)
                softmax =  torch.nn.Softmax(dim=1)
                outputs = softmax(model(images))
                pred.append(outputs)
                gtruth.append(target)
                id.append(batch_id)
            return zip(id, pred, gtruth)
    
    def find_predict(self, loader, model_id, full_targets):
        with torch.no_grad():
            pred = []
            id = []
            gtruth = []
            models_pred = self.models_predictions
            targets = np.array(full_targets)[loader.data[self.subset_name].indices]
            batch_targets = [targets[i:i + 32] for i in range(0, len(targets), 32)]
            batch_id = [loader.data[self.subset_name].indices[i:i + 32] for i in range(0, len(loader.data[self.subset_name].indices), 32)]
            for batch_id, target in zip(batch_id, batch_targets):
                outputs = [models_pred[model_id][i//32][i%32] for i in batch_id]
                pred.append(outputs)
                gtruth.append(target)
                id.append(batch_id)
            return zip(id, pred, gtruth)
    
    def fill_predictionCC(self, loader, model, resolution, threshold=0, targets = [], model_id = None):
        outputs = self.find_predict(loader=loader, model_id = model_id, full_targets = targets)                                 
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
                        resolution = resolution,
                        )
                    self.predictions.append(predcc)
        
    def global_predict(self):
        load = ImageNetDataLoader(root_dir = self.data_dir)
        self.list_id_unpredicted = [i for i in range(len(self.loaders[0].data[self.subset_name]))]
        targets = self.global_targets

        for threshold, model, resolution, loader, model_id in zip(self.thresholds, self.models[:-1], self.resolutions[:-1], self.loaders[:-1], [0,1]):
            loader = loader.select_data(list_id=self.list_id_unpredicted, subset_name = self.subset_name)

            self.fill_predictionCC(
                loader= loader, 
                model = model, 
                model_id = model_id,
                resolution = resolution, 
                threshold=threshold,
                targets = targets
                )
            
        model, resolution = self.models[-1], self.resolutions[-1]
        loader = self.loaders[-1]
        loader = loader.select_data(list_id=self.list_id_unpredicted, subset_name = self.subset_name)
        self.fill_predictionCC(
                loader=loader, 
                model= model,
                model_id = 2,
                resolution = resolution, 
                targets = targets
                )
                    
    def found_threshold(self, init_thresh = [0,0], step = [10,10], target_acc = None, r_dir = "."): #Recherche logarithmique 
        if init_thresh == None:
            init_thresh = [0 for i in range(dim)]
        self.list_id_unpredicted = None
        
        pre_pred = self.pre_pred()

        subset_name = "val"
        
        self.global_targets = [target for _, target in self.loaders[0].data[subset_name]]
        
        dim = len(self.models)
        r_time = []
        r_acc = []
        r_num_images = []    
        
        thresh_0 = np.linspace(init_thresh[0], 1, step[0])
        thresh_1 = np.linspace(init_thresh[1], 1, step[1])
        
        threshs = list(itertools.product(thresh_0, thresh_1))
        print(threshs)
        print(len(threshs))
        
        for thresh in threshs :
            
            cc = Cascade(self.data_dir, thresh, self.models, self.resolutions, targets = self.global_targets, loaders = self.loaders, pre_pred = pre_pred, subset_name = subset_name)

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
    
    def found_acc(self, targets_acc = [1], precision = 0.01 ,init_thresh = [0,0], final_thresh = [1, 1], step = 100, r_dir = "."): #So then plot FLOP // Acc +- 1%
        
        class out():
            def __init__(self, target):
                self.target = target
                self.flops = []
                self.thresh = None
                self.minflop = None
                self.acc = None
            def get_target(self):
                return self.target
            def update_min_flop(self):
                if self.flops != [] :
                    self.minflop = min(self.flops)
                return self.minflop
            def update_acc(self, acc):
                if self.flops[-1] == min(self.flops):
                    self.acc = acc
                return self.acc
            def update_thresh(self, thresh):
                if self.flops[-1] == min(self.flops):
                    self.thresh = thresh
                return self.acc
            
        output = [out(target=t) for t in targets_acc]
            
        thresh_0 = np.linspace(init_thresh[0], final_thresh[0], step[0])
        thresh_1 = np.linspace(init_thresh[1], final_thresh[1], step[1])
        threshs = list(itertools.product(thresh_0, thresh_1))
        pre_pred = self.pre_pred()
        out = []
        
        subset_name = "val"
        
        self.global_targets = [target for _, target in self.loaders[0].data[subset_name]]
             
        for thresh in threshs :
            cc = Cascade(self.data_dir, thresh, self.models, self.resolutions, targets = self.global_targets, loaders = self.loaders, pre_pred = pre_pred, subset_name = subset_name)
            cc.global_predict()
            acc = cc.stats.eval_acc()
            for out in output :
                if out.get_target() - precision < acc < out.get_target() + precision :
                    num_images = np.array(cc.stats.eval_resolutions())
                    prop_images = num_images/(len(self.loaders[0].data[self.subset_name]))
                    print("Props : {:.2f}, {:.2f}".format(prop_images[0], prop_images[1]))
                    flop = ResNetflop(prop_images)
                    out.flops.append(flop)
                    out.update_min_flop()
                    out.update_acc(acc)
                    out.update_thresh(thresh)
                    print("Number of flop : {:.2f}".format(flop))
                    print("Thresholds : ", thresh)
        list_acc = [] 
        list_flop = []
        list_thresh = []      
        for out in output :
            print("For acc :", out.get_target(), "Min flop is : " ,out.minflop, "Acc found: ", out.acc)
            list_acc.append(out.acc)
            list_flop.append(out.minflop)
            list_thresh.append(out.thresh)
        data_dict = {"flop": list_flop, "acc": list_acc, "thresh": list_thresh}
        with open(r_dir + '/accflop.json', "w") as json_file:
            json.dump(data_dict, json_file)
            
    def test_thresholds(self):
        list_acc = []
        list_flop = []
        global_targets = [target for _, target in self.loaders[0].data[self.subset_name]]
        
        with open('Results/accflop89.0_89.9.json', "r") as json_file:
            data_dict = json.load(json_file)
            
        pre_pred = self.pre_pred()

        threshs = data_dict["thresh"][:]
        for thresh in threshs:
            if thresh != None:
                cc = Cascade(self.data_dir, thresh, self.models, self.resolutions, targets = global_targets, loaders = self.loaders, pre_pred = pre_pred, subset_name="test")
                cc.global_predict()
                acc = cc.stats.eval_acc()
                list_acc.append(acc)
                
                num_images = np.array(cc.stats.eval_resolutions())
                print("Number images : {}".format(num_images))
                prop_images = num_images/(len(self.loaders[0].data["test"]))
                print("Props : {:.2f}, {:.2f}".format(prop_images[0], prop_images[1]))
                flop = ResNetflop(prop_images)
                list_flop.append(flop)
                print("Flops : {:.2f}".format(flop))
                
        data_dict = {"acc": list_acc, "flop": list_flop, "thresh": threshs}
        with open('Results/accfloptest.json', "w") as json_file:
            json.dump(data_dict, json_file)
                    
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
                #print("Number of elt predicted at resolution ", resolutions, " : ", count)
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
                val_loader = loader.data_batch[self.subset_name]
                acc = validate(val_loader, model, criterion, mute=True)
                print("Model{} Accuracy: {:.2f}%".format(i, acc))
