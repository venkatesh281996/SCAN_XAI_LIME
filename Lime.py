"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import torch
import yaml
import numpy as np
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader, get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt
import torchvision.transforms as T  # Importing transforms from torchvision
from IPython.display import display, Image as IPImage  # Import display module for Colab

import os

# --------Accepting Input----------
# Initialize parser
parser = argparse.ArgumentParser(description='Predict clusters within candidate images')
# Adding arguments
parser.add_argument("--n", required=True, help='Number of candidates')
parser.add_argument("--query", required=True, help='query image index')
parser.add_argument('--config_exp', help='Location of config file')
parser.add_argument('--model', help='Location where model is saved')
parser.add_argument('--visualize_prototypes', action='store_true', help='Show the prototype for each cluster')

args = parser.parse_args()

def main():
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512  # Ensure batch size fits on a single GPU

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_val_dataset(config, transforms)
    visualize_query_image(int(args.query), dataset)

    # Load nearest neighbors indices
    neighborhood_indices = np.load("/content/drive/MyDrive/ATiML-SCAN-Clustering-main/ATiML-SCAN-Clustering-main/repository_eccv/cifar-20/pretext/top"+args.n+"-val-neighbors.npy")
    indices = neighborhood_indices[int(args.query)][1:]
    candidate_data = [dataset.data[i] for i in indices]
    candidate_targets = [dataset.targets[i] for i in indices]
    dataset.data = candidate_data
    dataset.targets = candidate_targets

    dataloader = get_val_dataloader(config, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)

    # Load model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(args.model, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(state_dict)
    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])
    else:
        raise NotImplementedError

    # Ensure model is on CPU
    model = model.cpu()

    # Perform evaluation
    print(colored('Perform predict of the clustering model (setup={}).'.format(config['setup']), 'blue'))
    head = state_dict['head'] if config['setup'] == 'scan' else 0
    predictions, features = get_predictions(config, dataloader, model, return_features=True)
    clustering_stats = hungarian_evaluate(head, predictions, dataset.classes, compute_confusion_matrix=True)
    print(clustering_stats)

    # Organize clusters
    clusters = {}
    for i, p in enumerate(predictions[0]['predictions']):
        if p.item() not in clusters:
            clusters[p.item()] = [i]
        else:
            clusters[p.item()].append(i)
    for k in clusters:
        print(f"clusters {k}:") 
        for i in clusters[k]:
            print(i, end=",")
        print()

    # Visualize prototypes and LIME explanations
    if args.visualize_prototypes:
        prototype_indices_all_classes = get_prototypes(config, predictions[head], features, model)
        prototype_indices = [prototype_indices_all_classes[i] for i in sorted(list(clusters.keys()))]
        for idx in prototype_indices:
            print("Applying LIME on image:", idx)
            apply_lime_explainer(idx, dataset, model)


def get_prototypes(config, predictions, features, model, topk=1):
    import torch.nn.functional as F
    # Get topk most certain indices and pred labels
    print('Get topk')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]
    
    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim = 1)
    indices = torch.zeros((n_classes, topk), dtype=torch.long)
    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k=topk, largest=True, sorted=True)
        indices[pred_id, :] = conf_idx

    # Get corresponding features
    selected_features = torch.index_select(features, dim=0, index=indices.view(-1))
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    # Get mean feature per class
    mean_features = torch.mean(selected_features, dim=1)

    # Get min distance wrt to mean
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    # Get final indices
    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).bool()
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices

def apply_lime_explainer(idx, dataset, model):
    explainer = lime_image.LimeImageExplainer()
    
    # Get the image and preprocess it
    img = np.array(dataset.get_image(idx)).astype(np.uint8)
    original_image = img.copy()

    # Define a prediction function for LIME
    def predict_fn(images):
        model.eval()
        # Define transformations
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        images = [transform(Image.fromarray(image)) for image in images]
        images = torch.stack(images)
        outputs = model(images)
        if isinstance(outputs, list):
            outputs = outputs[0]
        return outputs.detach().cpu().numpy()

    # Apply LIME
    explanation = explainer.explain_instance(
        original_image,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Get the image and mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    # Overlay the LIME mask on the original image
    highlighted_image = mark_boundaries(temp / 255.0, mask)

    # Save and display the image with the LIME mask
    filename = f'lime_explanation_{idx}.png'
    plt.imshow(highlighted_image)
    plt.axis('off')
    plt.title(dataset[idx]['meta']['class_name'])
    plt.savefig(filename)
    plt.close()

    display(IPImage(filename))

def visualize_query_image(idx, dataset):
    print("Showing query image...")
    img = np.array(dataset.get_image(idx)).astype(np.uint8)
    img = Image.fromarray(img)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.title(dataset[idx]['meta']['class_name'])
    filename = f'query_image_{idx}.png'
    plt.savefig(filename)
    plt.close()

    display(IPImage(filename))


if __name__ == "__main__":
    main()
