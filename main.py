
import argparse
from transfert_learning import TransferLearningResNet
from data_loader import ImageNetDataLoader
import torch
import numpy as np
from cascade import Cascade

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--datadir', required=False, help='Path test and train data')
parser.add_argument('--model', type=str, help='Name of model to train')
parser.add_argument('--resolution', nargs='*', type=int, help='List of image sizes')

parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--batch', type=int, help='Batch size')
parser.add_argument('--lr', type=float, help='learning rate')

parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')

parser.add_argument('--training', action='store_true', help='Used to train before using cascade')
parser.add_argument('--cascade', action='store_true', help='Execute the cascade efficient')
parser.add_argument('--savepath', required=False, help='Path to save model')
parser.add_argument('--ccresults', required=False, help='Path to results save')
parser.add_argument('--modelsfolder',required=False, help='Path to model for casacade')


args = parser.parse_args()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    if args.cascade :
        model_path1 = args.modelsfolder + '/resnet18_resolution_32.pth'
        model_path2 = args.modelsfolder + '/resnet34_resolution_96.pth'
        model_path3 = args.modelsfolder + '/resnet50_resolution_224.pth'
        model1 = torch.load(model_path1)
        model2 = torch.load(model_path2)
        model3 = torch.load(model_path3)
        torch.cuda.empty_cache()
        cc = Cascade(data_dir=args.datadir, thresolds = None, models = [model1, model2, model3], resolutions = [32, 96, 224]) #Initialiation of the architecture
        #cc.found_threshold(init_thresh = [0.97, 0.74], step = [50,100], target_acc = None, r_dir = args.ccresults)

        #cc.found_acc(targets_acc = np.arange(72, 91, 0.5), precision = 0.1 , init_thresh = [0,0], final_thresh = [1,1], step = [40,40],  r_dir = args.ccresults)
        #cc.found_acc(targets_acc = np.arange(89.0, 90.5, 0.05), precision = 0.02 , init_thresh = [0.97,0.74], final_thresh = [1,1], step = [50,100],  r_dir = args.ccresults)
        cc.test_thresholds()
        
    if args.training :
        from transfert_learning import TransferLearningResNet
        dataset = ImageNetDataLoader(root_dir=args.datadir, batch_size=args.batch).get_data(resolution=args.resolution[0], pre_resize=False)
        dataset.load_data()
        net = TransferLearningResNet(args.model, epochs=args.epochs, loader=dataset, path_to_save=args.savepath, lr=args.lr)

if __name__=='__main__':
   main()