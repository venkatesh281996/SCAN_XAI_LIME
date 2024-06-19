"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
import numpy as np
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from PIL import Image
from copy import deepcopy


# --------Accepting Input----------
# Initialize parser
parser = argparse.ArgumentParser(description='Predict clusters within candidate images')
# Adding arguments
parser.add_argument("--query", required=True, help='query image index')
parser.add_argument('--config_exp', help='Location of config file')
parser.add_argument('--model', help='Location where model is saved')
parser.add_argument('--visualize_prototypes', action='store_true', 
                    help='Show the prototpye for each cluster')

args = parser.parse_args()

def main():
    
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_val_dataset(config, transforms)
    visualize_query_image(int(args.query), dataset)
    
    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(args.model, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(state_dict)

    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])

    else:
        raise NotImplementedError
        
    # CUDA
    model.cuda()

    # Perform evaluation
    print(colored('Perform predict of the clustering model (setup={}).'.format(config['setup']), 'blue'))
    head = state_dict['head'] if config['setup'] == 'scan' else 0

    acc_list = []
    ari_list = []
    nmi_list = []
    top5_list = []
    number_of_candidates = list(range(50,1001,50))

    for n in number_of_candidates:
        new_dataset = deepcopy(dataset)
        neighborhood_indices = np.load("repository_eccv\\cifar-20\\pretext\\top"+str(n)+"-val-neighbors.npy")
        indices = neighborhood_indices[int(args.query)][1:]
        candidate_data = [dataset.data[i] for i in indices]
        candidate_targets = [dataset.targets[i] for i in indices]
        new_dataset.data = candidate_data
        new_dataset.targets = candidate_targets

        dataloader = get_val_dataloader(config, new_dataset)

        print('Number of samples: {}'.format(len(new_dataset)))

        print("Getting performance metrics of clustering {} candidates".format(len(new_dataset)))
        predictions, features = get_predictions(config, dataloader, model, return_features=True)
        clustering_stats = hungarian_evaluate(head, predictions, new_dataset.classes, 
                                                compute_confusion_matrix=False)
        acc_list.append(clustering_stats['ACC'])
        ari_list.append(clustering_stats['ARI'])
        nmi_list.append(clustering_stats['NMI'])
        top5_list.append(clustering_stats['ACC Top-5'])

    metrics = ['ACC', 'ARI', 'NMI', 'ACC Top-5']
    for m in metrics:
        if m=="ACC":
            visualize_clustering_stats(number_of_candidates, acc_list, "Accuracy")
        if m=="ARI":
            visualize_clustering_stats(number_of_candidates, ari_list, "Adjusted Rand Index")
        if m=="NMI":
            visualize_clustering_stats(number_of_candidates, nmi_list, "Normalized Mutual Information")
        if m=="ACC Top-5":
            visualize_clustering_stats(number_of_candidates, top5_list, "Accuracy with Top 5 neighbors")

def visualize_query_image(idx, dataset):
    import matplotlib.pyplot as plt
    import numpy as np

    img = np.array(dataset.get_image(idx)).astype(np.uint8)
    img = Image.fromarray(img)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def visualize_clustering_stats(x, y, metric):
    import matplotlib.pyplot as plt

    plt.xlabel('Number of Candidates')
    plt.ylabel(metric)
    plt.title(metric+' Evaluation')

    plt.grid(True)
    plt.plot(x, y)

    plt.show()

if __name__ == "__main__":
    main()
