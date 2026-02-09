import argparse
import ast
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.conf_hml_loss import ConfHMLLossWithProbs
from utils.echinoderms_dataset import EchinodermsDataset
from utils.multihead_utils import build_multi_headed_model
from utils.utils import set_seed
from utils.vision_model_utils import convert_data_to_bin,get_benthic_node_freqs, get_balanced_weights_echinoderms, set_requires_grad

# Puppet class for train and test
class PuppetDataset():
    def __init__(self):
        self.to_eval = torch.Tensor([0,1,1,1,1,1,1,1,1,1,1,1,1])


def gen_echinoderms_focal_args():
    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--backbone', type=str, default="vit_b_16",)
    parser.add_argument('--freeze_enc', type=bool, default=False,)
    parser.add_argument('--focal_k', type=float, default=1.0,
                        help='Focal k parameter (default: 1.0)')
    parser.add_argument('--focal_min', type=float, default=0.25,
                        help='Focal min parameter (default: 0.25)')
    parser.add_argument('--focal_mode', type=str, default="mean",)
    parser.add_argument('--use_backbone_ckpt', type=bool, default=False,
                        help='Whether to use a pre-trained backbone checkpoint (default: False)')
    parser.add_argument('--use_node_weighting', type=bool, default=False,
                        help='Whether to use node weighting (default: False)')
    parser.add_argument('--noise_factor', type=float, default=0.0,
                        help='Noise factor for mixing encoder weights (default: 0.0)')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = gen_echinoderms_focal_args()

    print("Arguments:", args)
    
    seed = args.seed
    mix_enc_noise = args.noise_factor

    backbone = args.backbone # resnet50 or vit_b_16
    # Backbone
    if args.use_backbone_ckpt:
        if args.backbone == "resnet50":
            backbone_checkpoint_path = "./pre-trained_models/bt-600e-prod_rn50_epoch=599.ckpt"
        elif args.backbone == "vit_b_16":
            backbone_checkpoint_path = "./pre-trained_models/mocov3-100e-prod_vit-b_epoch=099.ckpt"
    else:
        backbone_checkpoint_path = None

    # Node weighting
    node_weighting = True

    # Training parameters
    lr = 1e-3
    weight_decay = 1e-4
    num_epochs = 20
    target_omega = 0.25

    freeze_enc = args.freeze_enc

    batch_size = 128
    data_key = "echinoderms"
    n_nodes = 13 # There are 13 nodes in the echinoderm hierarchy

    image_dir = ""

    # Multihead parameters
    focal_k = args.focal_k
    focal_min = args.focal_min
    focal_mode = args.focal_mode # "aleatoric, "epistemic", "predictive", or "pcs"
    n_heads = 20

    set_seed(seed, input_args=False)

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

    # Prepare data normalization
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

    hyperparams = {
        "num_layers": 3,
        "dropout": 0.7,
        "non_lin": "relu",
    }

    # Prepare data
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
        local=image_dir,
        transform=transform,
    )

    test_echinoderm_data = EchinodermsDataset(
        annotations=test_df,
        local=image_dir,
        transform=transform,
    )


    dataloader = DataLoader(echinoderm_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_echinoderm_data, batch_size=batch_size, shuffle=False, drop_last=False)


    # Prepare model
    R = R.to(device)

    model = build_multi_headed_model(
            n_heads=n_heads,
            data_str=data_key,
            hyperparams=hyperparams,
            backbone=backbone,
            backbone_checkpoint_path=backbone_checkpoint_path,
            R=R,
            n_nodes=n_nodes,
            freeze_enc=freeze_enc,
            mix_enc_noise=mix_enc_noise,
    )

    model = model.to(device)

    # Prepare eval masks
    train = PuppetDataset()
    test = PuppetDataset()

    train_eval_index=train.to_eval.bool()
    test_eval_index=test.to_eval.bool()

    # Prepare optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Prepare confidence-based focal loss
    conf_hml_loss = ConfHMLLossWithProbs(
        k=focal_k,
        R=R, 
        weights=weights,
        focal_min=focal_min,
        train_eval_index=train_eval_index,
        mode=focal_mode,
        use_node_weighting=args.use_node_weighting,
    )

    conf_hml_loss = conf_hml_loss.to(device)

    # Training Loop
    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        mean_loss, mean_acc, mean_unc, ap, bin_ap, f1, precision, recall = model.fit(
            train_loader=dataloader,
            optimizer=optimizer,
            conf_hml_loss=conf_hml_loss,
            device=device,
        )

        print(f"\tMean loss: {mean_loss}, Mean accuracy: {mean_acc}, Mean confidence: {1-mean_unc}")
        print(f"\tMean optimistic AP: {ap}, Mean bin AP: {bin_ap}")
        print(f"\tMean F1: {f1}, Mean precision: {precision}, Mean recall: {recall}")

    # Testing
    model.test(
        eval_loader=test_dataloader ,
        test_eval_index=test_eval_index,
        conf_hml_loss=conf_hml_loss,
        seed=seed,
        num_epochs=num_epochs,
        mix_enc_noise=mix_enc_noise,
        num_heads=n_heads,
        focal_k=focal_k,
        focal_min=focal_min,
        dataset_name=data_key+"_"+backbone+"_f-"+str(freeze_enc),
        conf_method=focal_mode,
        write_to_file=True,
        device=device
    )