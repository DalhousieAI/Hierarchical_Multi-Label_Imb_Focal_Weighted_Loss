# Code adpated from:
# https://github.com/DalhousieAI/benthicnet_probes/blob/master/utils/utils.py
# Under GPL-3.0 License

import numpy as np
import torch

_VIT_NUM_FEATURES = {
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
}

def load_model_state(model, ckpt_path, origin="", component="encoder", verbose=0):
    # key = 'state_dict' for pre-trained models, 'model' for FB Imagenet
    alt_component_names = {
        "encoder": "backbone",
    }
    alt_component_name = alt_component_names.get(component, "")

    loaded_dict = torch.load(ckpt_path)

    if origin == "fb":
        key = "model"
    else:
        key = "state_dict"

    state = loaded_dict[key]
    loading_state = {}
    model_keys = model.state_dict().keys()

    if any(s in ckpt_path for s in ("mocov3", "mae", "vit")) and component == "encoder":
        if any(s in ckpt_path for s in ("hp", "hl", "hft")):
            loading_state = get_vit_state(
                model, state, model_keys, loading_state, reorder_pos_emb=False
            )
        else:
            loading_state = get_vit_state(model, state, model_keys, loading_state)
    else:
        for k in list(state.keys()):
            k_split = k.split(".")
            k_0 = k_split[0]
            if len(k_split) > 1:
                k_1 = k_split[1]
            else:
                k_1 = ""

            k_heads = ".".join([k_0, k_1])
            if k_0 == component or k_heads == component:
                k_to_check = k.replace(f"{component}.", "")
            elif k_0 == alt_component_name or k_heads == alt_component_name:
                k_to_check = k.replace(f"{alt_component_name}.", "")
            else:
                k_to_check = k

            if k_to_check in model_keys:
                loading_state[k_to_check] = state[k]
    if verbose > 0:
        print(
            f"Loading {len(loading_state.keys())} layers for {component}\n"
            " Expected layers (approx):\n\tViT-Base: 150\n\tViT-Large: 294\n\tResNet-50: 320"
        )
    model.load_state_dict(loading_state, strict=False)
    if verbose > 0:
        print(f"Loaded {component} from {ckpt_path}.")

    return model


# Function for supporting ViT loading
def get_vit_state(model, state, model_keys, loading_state, reorder_pos_emb=True):
    # Remove default ImageNet head from requiring loading
    model_keys = list(model_keys)[:-2]
    state_list = list(state.items())
    if reorder_pos_emb:
        pos_emb = state_list[1]
        conv_proj_w = state_list[2]
        conv_proj_b = state_list[3]

        state_list[1] = conv_proj_w
        state_list[2] = conv_proj_b
        state_list[3] = pos_emb

    for i, key in enumerate(model_keys):
        try:
            assert model.state_dict()[key].shape == state_list[i][1].shape
            loading_state[key] = state_list[i][1]
        except AssertionError:
            print(
                f"\nViT layer {i} {key}, does not match loading state layer {state_list[i][0]}"
            )
            print(
                f"Expected shape: {model.state_dict()[key].shape}, "
                f"from loading state got shape: {state_list[i][1].shape}"
            )
            continue
    return loading_state

# Freeze model weights
def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val

def convert_idx_to_bin(row, n_nodes, col="catami_biota"):
    bin_notation = np.zeros(n_nodes)
    idxs = row[col]
    for idx in idxs:
        bin_notation[idx] = 1
    
    return bin_notation

def convert_data_to_bin(data, n_nodes, col="catami_biota"):
    data[col] = data.apply(lambda x: convert_idx_to_bin(x, n_nodes, col), axis=1)
    return data

def get_benthic_node_freqs(data, R):
    # Obtain the frequency of each node in the training set
    node_frequencies = data.sum(axis=0)

    if R is not None:
        for i in range(len(node_frequencies)):
            descendent_idxs = np.where(R[0][i])[0]
            node_frequencies[i] = node_frequencies[descendent_idxs].sum()
    
    dataset_len = len(data)

    return node_frequencies, dataset_len

def get_balanced_weights_echinoderms(
        node_frequencies, 
        dataset_len, 
        mode="class-wise", 
        min_weight_const=0.5,
        ):
    if mode == "binary":
        total_classes = 2 # Since binary classification for each node
    elif mode == "class-wise":
        total_classes = node_frequencies.shape[0]
    else:
        raise ValueError("Invalid mode. Only 'binary' and 'class-wise' are supported.")

    def calc_balanced_weights(node_frequencies, total_classes, dataset_len):
        # Zero and maximum frequency cases
        zero_freq_idxs = np.where(node_frequencies == 0)[0]
        max_freq_idxs = np.where(node_frequencies == dataset_len)[0]

        node_weights = dataset_len/(total_classes*node_frequencies)

        # Set the weights of the nodes with zero and max frequency (always on) to 1
        node_weights[zero_freq_idxs] = 1
        node_weights[max_freq_idxs] = 1

        return node_weights

    positive_node_weights = calc_balanced_weights(node_frequencies, total_classes, dataset_len)
    # Scale positive weights to be between 0 and 1
    min_pos_weight = positive_node_weights.min()
    max_pos_weight = positive_node_weights.max()

    positive_node_weights = min_weight_const + max_pos_weight * (positive_node_weights - min_pos_weight) / (max_pos_weight - min_pos_weight)
    
    return positive_node_weights
