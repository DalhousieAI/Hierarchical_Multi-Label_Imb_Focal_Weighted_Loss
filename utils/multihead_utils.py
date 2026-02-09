from xml.parsers.expat import model
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ResNet50_Weights

from utils.constrained_ff import ConstrainedFFNNModel
from utils.multihead_definitions import MultiHeadedModel
from utils.utils import input_dims, output_dims, hidden_dims
from utils.vision_model_utils import load_model_state, _VIT_NUM_FEATURES, set_requires_grad

import torch
from torchvision import models

def mix_enc_weights(enc, backbone, noise_factor):
    # Get current (pretrained) weights
    pretrained_state_dict = enc.state_dict()

    # Create a new instance of the same model, randomly initialized
    if backbone == "resnet50":
        random_model = models.resnet50(weights=None)
    elif backbone == "vit_b_16":
        random_model = models.vit_b_16(weights=None)
    else:
        raise ValueError(f"Backbone {backbone} not supported for weight mixing")
    
    random_state_dict = random_model.state_dict()

    # Interpolate weights
    mixed_state_dict = {}
    for key in pretrained_state_dict:
        pre = pretrained_state_dict[key]
        rand = random_state_dict[key]
        # Mix the weights
        mixed = (1 - noise_factor) * pre + noise_factor * rand
        mixed_state_dict[key] = mixed

    # Load the mixed weights into the model
    enc.load_state_dict(mixed_state_dict)
    return enc

def build_multi_headed_model(
        n_heads,
        data_str,
        backbone,
        backbone_checkpoint_path,
        R,
        hyperparams=None,
        ontology=None,
        num_to_skip=None,
        n_nodes=None,
        freeze_enc=False,
        mix_enc_noise=0.0,
):
        enc = None

        if backbone is not None:
                assert n_nodes is not None, "n_nodes must be specified if backbone is used"
                if backbone == "resnet50":
                        enc = models.resnet50(weights="DEFAULT")
                elif backbone == "vit_b_16":
                        enc = models.vit_b_16(weights="DEFAULT")
        
                if mix_enc_noise > 0.0:
                        enc = mix_enc_weights(enc, backbone, mix_enc_noise)

                # Remove the last layer
                if "resnet" in backbone:
                        features_dim = enc.inplanes
                        enc.fc = nn.Identity()
                elif "vit" in backbone:
                        features_dim = _VIT_NUM_FEATURES[backbone]
                        enc.heads = nn.Identity()
                else:
                        print("No adjusment to:", backbone)

                if backbone_checkpoint_path is not None:
                        enc = load_model_state(
                                model=enc,
                                ckpt_path=backbone_checkpoint_path,
                        )

                input_dim = features_dim
                emb_dim = features_dim
                output_dim = n_nodes

                if freeze_enc:
                        set_requires_grad(enc, False)

        else:
                input_dim = input_dims[data_str]
                emb_dim = hidden_dims[ontology][data_str]
                output_dim = output_dims[ontology][data_str] + num_to_skip

        heads = []
        for _ in range(n_heads):
                head = ConstrainedFFNNModel(
                                input_dim=input_dim, 
                                hidden_dim=emb_dim, 
                                output_dim=output_dim,
                                hyperparams=hyperparams,
                                R=R
                        )
                heads.append(
                        head
                        )
        heads = nn.ModuleList(heads)
        model = MultiHeadedModel(enc=enc, heads=heads)

        return model