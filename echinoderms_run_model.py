
import argparse
import ast
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.constrained_ff import ConstrainedFFNNModel
from utils.utils import set_seed, train_model
from utils.vision_model_utils import load_model_state, _VIT_NUM_FEATURES
from utils.echinoderms_dataset import EchinodermsDataset
from utils.vision_model_utils import convert_data_to_bin,get_benthic_node_freqs, get_balanced_weights_echinoderms, set_requires_grad

def gen_echinoderms_args():
    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--use_backbone_ckpt', type=bool, default=False,
                        help='Whether to use a pre-trained backbone checkpoint (default: True)')
    parser.add_argument('--target_omega', type=float, default=0.25,
                        help='target omega for the node weighting (default: 0.0)')
    parser.add_argument('--backbone', type=str, default="vit_b_16",)
    parser.add_argument('--freeze_enc', type=bool, default=False,)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = gen_echinoderms_args()

    print("Arguments:", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set cuda blocking for debugging
    seed = args.seed

    # Backbone
    backbone = args.backbone # resnet50 or vit_b_16

    if backbone == "resnet50":
        backbone_checkpoint_path = "./pre-trained_models/bt-600e-prod_rn50_epoch=599.ckpt"
    elif backbone == "vit_b_16":
        backbone_checkpoint_path = "./pre-trained_models/mocov3-100e-prod_vit-b_epoch=099.ckpt"

    # Node weighting
    if args.target_omega >= 0.0:
        node_weighting = True
    else:
        node_weighting = False

    # Training parameters
    image_dir = ""
    lr = 1e-3
    num_epochs = 20
    weight_decay = 1e-4
    target_omega = args.target_omega

    freeze_enc = args.freeze_enc
        
    batch_size = 128
    n_nodes = 13 # There are 13 nodes in the echinoderm hierarchy

    # Echinoderm hierarchy
    R = [
            [
                [1,1,1,1,1,1,1,1,1,1,1,1,1,],
                [0,1,0,0,0,0,1,0,0,0,0,0,0,],
                [0,0,1,0,0,0,0,1,1,0,0,0,0,],
                [0,0,0,1,0,0,0,0,0,1,1,0,0,],
                [0,0,0,0,1,0,0,0,0,0,0,0,0,],
                [0,0,0,0,0,1,0,0,0,0,0,1,1,],
                [0,0,0,0,0,0,1,0,0,0,0,0,0,],
                [0,0,0,0,0,0,0,1,0,0,0,0,0,],
                [0,0,0,0,0,0,0,0,1,0,0,0,0,],
                [0,0,0,0,0,0,0,0,0,1,0,0,0,],
                [0,0,0,0,0,0,0,0,0,0,1,0,0,],
                [0,0,0,0,0,0,0,0,0,0,0,1,0,],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,],
        ]
    ]

    R = torch.Tensor(R)

    set_seed(seed, input_args=False)    

    benthicnet_mean_std = transforms.Normalize(
            mean=[0.359, 0.413, 0.386],
            std=[0.219, 0.215, 0.209],
        )

    transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                benthicnet_mean_std,
            ]
        )

    data_df = pd.read_csv("./HMC_data/echinoderms/echinoderms.csv")
    data_df["catami_biota"] = data_df["catami_biota"].apply(ast.literal_eval)

    data_df = convert_data_to_bin(data=data_df, n_nodes=n_nodes)

    train_df = data_df[data_df["partition"] == "train"]
    print("Training Data:", len(train_df))

    if node_weighting:
        imb_method = "class-wise"
        node_frequencies, train_len = get_benthic_node_freqs(train_df["catami_biota"], R)
        pos_weights = get_balanced_weights_echinoderms(
            node_frequencies=node_frequencies,
            dataset_len=train_len,
            min_weight_const=target_omega
        )
        neg_weights = np.ones(pos_weights.shape)
        weights = np.array([pos_weights, neg_weights])
    else:
        imb_method = "none"
        weights = None

    test_df = data_df[data_df["partition"] == "test"]
    print("Testing Data:", len(test_df))

    echinoderm_data = EchinodermsDataset(
        annotations=train_df,
        transform=transform,
        local=image_dir,
    )

    test_echinoderm_data = EchinodermsDataset(
        annotations=test_df,
        transform=transform,
        local=image_dir,
    )

    dataloader = DataLoader(echinoderm_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_echinoderm_data, batch_size=batch_size, shuffle=False)

    if backbone == "resnet50":
        enc = models.resnet50(weights="DEFAULT")
    elif backbone == "vit_b_16":
        enc = models.vit_b_16(weights="DEFAULT")

    # Remove the last layer
    if "resnet" in backbone:
        features_dim = enc.inplanes
        enc.fc = nn.Identity()
    elif "vit" in backbone:
        features_dim = _VIT_NUM_FEATURES[backbone]
        enc.heads = nn.Identity()
    else:
        print("No adjusment to:", backbone)

    if args.use_backbone_ckpt:
        enc = load_model_state(
            model=enc,
            ckpt_path=backbone_checkpoint_path,
        )

    if freeze_enc:
        set_requires_grad(enc, False)

    R = R.to(device)

    hyperparams = {
        "num_layers": 3,
        "dropout": 0.7,
        "non_lin": "relu",
    }

    head = ConstrainedFFNNModel(
        input_dim = features_dim,
        hidden_dim = features_dim,
        output_dim=n_nodes,
        hyperparams=hyperparams,
        R=R
    )

    class EchinodermModel(nn.Module):
        def __init__(self, enc, head):
            super(EchinodermModel, self).__init__()
            self.enc = enc
            self.head = head

        def forward(self, x):
            x = self.enc(x)
            x = self.head(x)
            return x
    
    model = EchinodermModel(enc, head)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Puppet class for train and test
    class PuppetDataset():
        def __init__(self):
            self.to_eval = torch.Tensor([0,1,1,1,1,1,1,1,1,1,1,1,1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = PuppetDataset()
    test = PuppetDataset()

    model = model.to(device)

    train_model(
            num_epochs=num_epochs, 
            model=model, 
            train_loader=dataloader, 
            test_loader=test_dataloader, 
            device=device, 
            optimizer=optimizer, 
            R=R, 
            train=train, 
            test=test, 
            dataset_name="echinoderms_CATAMI_rn50_from_init", 
            seed=seed, 
            imb_method=imb_method,
            mixed_loss_lambda=None,
            weights=weights,
            target_omega=target_omega,
            )