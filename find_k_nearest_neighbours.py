"""
Authors: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import numpy as np
import torch

from utils.config import create_config
from utils.common_config import get_model, get_train_dataset, \
                                get_val_dataset, \
                                get_val_dataloader, \
                                get_val_transformations
                                
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

# Uninstall any existing installations of faiss

import faiss

# Parser
parser = argparse.ArgumentParser(description='Eval_nn')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # Model
    model = get_model(p)
    model = model.cuda()
   
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} val samples'.format(len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Checkpoint
    checkpoint_path = '/content/drive/MyDrive/ATiML-SCAN-Clustering-main/ATiML-SCAN-Clustering-main/repository_eccv/cifar-20/selflabel/model.pth.tar'
    print(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    print(colored('Restart from checkpoint {}'.format(checkpoint_path), 'blue'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Fix key mismatch by filtering out mismatched keys
    model_dict = model.state_dict()
    checkpoint_dict = {k.replace('cluster_head', 'contrastive_head'): v for k, v in checkpoint.items() if k.replace('cluster_head', 'contrastive_head') in model_dict and v.size() == model_dict[k.replace('cluster_head', 'contrastive_head')].size()}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    
    model.cuda()

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    number_of_candidate_images = list(range(50, 1001, 50))
    for no_of_cand_images in number_of_candidate_images:
        print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
        fill_memory_bank(val_dataloader, model, memory_bank_val)
        topk = no_of_cand_images
        print('Mine the nearest neighbors (Top-%d)' % (topk))

        # Initialize GPU resources and create FAISS index on GPU
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(memory_bank_val.features.size(1))  # Create CPU index
        index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
        
        indices, acc = memory_bank_val.mine_nearest_neighbors(topk, index)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' % (topk, 100 * acc))
        np.save("repository_eccv/cifar-20/pretext/top" + str(no_of_cand_images) + "-val-neighbors.npy", indices)   

if __name__ == '__main__':
    main()
