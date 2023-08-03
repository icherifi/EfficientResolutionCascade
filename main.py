import argparse
from transfert_learning import TransferLearningResNet
from data_loader import ImageNetDataLoader
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--datadir', type=str, help='Path to trainning and test data')
parser.add_argument('--model', type=str, help='Name of model to train')
parser.add_argument('--resolution', nargs='*', type=int, help='List of image sizes')

parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--batch', type=int, help='Batch size')

parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')

parser.add_argument('--training', action='store_true', help='Used to train before using cascade')
parser.add_argument('--cascade', action='store_true', help='Execute the cascade efficient')
parser.add_argument('--savepath', required=False, help='Path to save model')


args = parser.parse_args()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    if args.cascade :
        model = "resnet18"
        loader = ImageNetDataLoader(args.datadir, args.batch)
        net = TransferLearningResNet(args.model, loader, args.epochs, path_to_model=args.ckpt)
    if args.training :
        from transfert_learning import TransferLearningResNet
        dataset = ImageNetDataLoader(args.datadir).get_data(resolution=args.resolution[0])
        loader = dataset.load_data()
        net = TransferLearningResNet(args.model, epochs=args.epochs, loader=dataset, path_to_save=args.savepath)

if __name__=='__main__':
   main()