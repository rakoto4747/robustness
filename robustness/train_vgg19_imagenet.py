
from argparse import ArgumentParser
import os, fnmatch
from . import model_utils, datasets, train, defaults
from .datasets import CIFAR, ImageNet

import torch as ch
from cox.utils import Parameters
import cox.store


def find_latest_checkpoint(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
      for name in files:
        if fnmatch.fnmatch(name, pattern):
          result.append(os.path.join(root, name))
    if result:
      result = sorted(result, reverse=True)[0]
    else:
      result = None
    return result
  
parser = argparse.ArgumentParser()
parser.add_argument("--imagenet-path", help="path to the ImageNet dataset in ImageFolder-readable format", type=str)
args = parser.parse_args()

DATASET_PATH= args.imagenet_path #ImageNet path
OUT_DIR = 'log' # directory to store log and checkpoints
NUM_WORKERS = 32
BATCH_SIZE = 256
CHECKPOINT_PATH = find_latest_checkpoint('*.pt', OUT_DIR) # latest checkpoint

def main():
    #Dataset
    dataset = CIFAR(DATASET_PATH)
    #Model
    model, checkpoint = model_utils.make_and_restore_model(arch='vgg19', dataset=dataset, 
                                                           resume_path=CHECKPOINT_PATH)
    #DataLoader
    train_loader, val_loader = dataset.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS)
    if 'module' in dir(model): model = model.module
    # Create a cox store for logging
    out_store = cox.store.Store(OUT_DIR)
    # Hard-coded base parameters
    train_kwargs = {
        'out_dir': "train_out",
        'adv_train': 1,
        'constraint': '2',
        'eps': 3.0,
        'attack_lr': 0.5,
        'attack_steps': 7,
        'save_ckpt_iters': 5
    }
    train_args = Parameters(train_kwargs)
    # Fill whatever parameters are missing from the defaults
    train_args = defaults.check_and_fill_args(train_args,
                            defaults.TRAINING_ARGS, CIFAR)
    train_args = defaults.check_and_fill_args(train_args,
                            defaults.PGD_ARGS, CIFAR)
    # Train a model
    train.train_model(train_args, model, (train_loader, val_loader), store=out_store, 
                    checkpoint=checkpoint)   

                  
if __name__ == "__main__":
    main()
