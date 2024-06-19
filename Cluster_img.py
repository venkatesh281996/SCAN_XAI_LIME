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

import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import display, Image as imgdisplay

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --------Accepting Input----------
# Initialize parser
parser = argparse.ArgumentParser(description='Predict clusters within candidate images')
# Adding arguments
parser.add_argument("--n", required=True, help='Number of candidates')
parser.add_argument("--query", required=True, help='Query image index')
parser.add_argument('--config_exp', help='Location of config file')
parser.add_argument('--model', help='Location where model is saved')
parser.add_argument('--visualize_prototypes', action='store_true', help='Show the prototype for each cluster')

args = parser.parse_args()

def main():
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512  # To make sure we can evaluate on a single 1080ti

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_val_dataset(config, transforms)
    visualize_query_image(int(args.query), dataset)
    neighborhood_indices = np.load("repository_eccv/cifar-20/pretext/top" + args.n + "-val-neighbors.npy")
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
    model.cpu()

    # Perform evaluation
    print(colored('Perform predict of the clustering model (setup={}).'.format(config['setup']), 'blue'))
    head = state_dict['head'] if config['setup'] == 'scan' else 0
    predictions, features = get_predictions(config, dataloader, model, return_features=True)
    clustering_stats = hungarian_evaluate(head, predictions, dataset.classes, compute_confusion_matrix=True)
    print(clustering_stats)
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
    if args.visualize_prototypes:
        prototype_indices_all_classes = get_prototypes(config, predictions[head], features, model)
        prototype_indices = [prototype_indices_all_classes[i] for i in sorted(list(clusters.keys()))]
        visualize_indices(prototype_indices, dataset, clustering_stats['hungarian_match'])
        visualize_query_image(int(args.query), dataset)

def grad_cam_call(idx, dataset):
    model_builder = keras.applications.xception.Xception
    img_size = (299, 299)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    last_conv_layer_name = "block14_sepconv2_act"

    img = np.array(dataset.get_image(idx)).astype(np.uint8)
    image = Image.fromarray(img)
    image = image.resize(size=img_size)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axes
    plt.title(dataset[idx]['meta']['class_name'])
    plt.show()


    array = np.expand_dims(image, axis=0)
    img_array = preprocess_input(array)

    # Make model
    model = model_builder(weights="imagenet")

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    preds = model.predict(img_array)
    print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    save_and_display_gradcam(image, heatmap, dataset[idx]['meta']['class_name'],idx)

def save_and_display_gradcam(img_object, heatmap, title,idx, cam_path="cam.jpg", alpha=0.4):
    img = keras.utils.img_to_array(img_object)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    image = mpimg.imread(cam_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axes
    plt.title(title)
    plt.show()
    path = "/content/drive/MyDrive/ATiML-SCAN-Clustering-main/ATiML-SCAN-Clustering-main/data/images/"
    ing = f"{idx}.jpg"
    newpath = path + ing
    superimposed_img.save(newpath)

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@torch.no_grad()
def get_prototypes(config, predictions, features, model, topk=1):
    import torch.nn.functional as F
    print('Get topk')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]

    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim=1)
    indices = torch.zeros((n_classes, topk))
    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k=topk, largest=True, sorted=True)
        indices[pred_id, :] = conf_idx

    selected_features = torch.index_select(features, dim=0, index=indices.view(-1).long())
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    mean_features = torch.mean(selected_features, dim=1)
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).bool()  # Convert to boolean tensor
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices

def visualize_indices(indices, dataset, hungarian_match):
    for idx in indices:
        print("Executing grad-cam on image:", idx)
        grad_cam_call(idx, dataset)
        plt.show()  # Ensure that the plot is shown in Colab

def visualize_query_image(idx, dataset):
    print("Showing query image...")
    img = np.array(dataset.get_image(idx)).astype(np.uint8)
    img = Image.fromarray(img)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.title(dataset[idx]['meta']['class_name'])
    plt.show()


if __name__ == "__main__":
    main()
