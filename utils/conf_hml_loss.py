import torch

from torch import nn
import torch.nn.functional as F
from utils.constrained_ff import get_constr_out, get_constr_out_vectorized

class ConfHMLLossWithProbs(nn.Module):
    def __init__(
            self, 
            k, 
            R, 
            weights, 
            focal_min, 
            train_eval_index, 
            mode,
            use_node_weighting=True,
            ):
        super(ConfHMLLossWithProbs, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.k = k
        self.R = R
        self.weights = weights
        self.focal_min = focal_min
        self.train_eval_index = train_eval_index
        self.mode = mode
        self.use_node_weighting = use_node_weighting

    def forward(self, probs, labels):
        losses = []
        predictions = []

        labels = labels

        processed_weights = process_weights(
                self.weights, 
                labels, 
                self.train_eval_index, 
                labels.device
            )
        if not self.use_node_weighting:
            processed_weights = torch.ones_like(processed_weights)
        
        uncertainties = calc_uncertainty(probs, self.mode)

        losses, predictions = vectorized_loss(
            probs, labels, processed_weights, self.R, self.train_eval_index
        )

        uncertainties = uncertainties.unsqueeze(dim=0)
        uncertainties = uncertainties[:, :, self.train_eval_index]
        uncertainties = uncertainties.expand_as(losses)
        
        losses = losses*(self.focal_min + uncertainties**self.k)
        losses = losses.sum()
        
        mean_predictions = predictions.mean(dim=0)

        return losses, mean_predictions, predictions, uncertainties[0]

def process_weights(
        weights, 
        labels,
        train_eval_index,
        device, 
        ):
    
    applied_weights = torch.Tensor(weights).to(device).T[train_eval_index]

    # Turn weights from (n_nodes, 2) to (batch_size, n_nodes, 2)
    applied_weights = applied_weights.unsqueeze(0).expand(labels.size(0), -1, -1)

    # Use targets as mask for applied weights such that the end result is a (batch_size, n_nodes) tensor
    weight_mask = labels[:, train_eval_index].int()

    rows = torch.arange(applied_weights.shape[0], device=applied_weights.device).view(-1, 1).expand(-1, applied_weights.shape[1])
    cols = torch.arange(applied_weights.shape[1], device=applied_weights.device).view(1, -1).expand(applied_weights.shape[0], -1)

    applied_weights = applied_weights[rows, cols, weight_mask]

    return applied_weights.to(device)

def calc_kl_div(p, q):
    kl_div = p * (torch.log2(p + 1e-10) - torch.log2(q + 1e-10)) + \
        (1 - p) * (torch.log2(1 - p + 1e-10) - torch.log2(1 - q + 1e-10))
    kl_div = torch.clamp(kl_div, min=0)
    return kl_div

def calc_js_div(p, q):
    m = 0.5 * (p + q)
    js_div = 0.5 * (calc_kl_div(p, m) + calc_kl_div(q, m))
    js_div = torch.clamp(js_div, min=0)
    return js_div

def calc_entropy(
        probs,
):
    # Calculate entropy for each sample using log baes 2
    entropy_pos = -probs * torch.log2(probs + 1e-10)
    entropy_neg = -(1 - probs) * torch.log2(1 - probs + 1e-10)

    entropy = entropy_pos + entropy_neg

    return entropy

def calc_epistemic_uncertainty(
        probs,
        metric='js_div',
):
    # shape [H, 1, B, C]
    probs_a = probs.unsqueeze(1)
    # shape [1, H, B, C]
    probs_b = probs.unsqueeze(0)

    if metric == 'kl_div':
        epistemic_uncertainties = calc_kl_div(probs_a, probs_b)
    else:
        epistemic_uncertainties = calc_js_div(probs_a, probs_b)

    mean_epistemic_uncertainties_0 = epistemic_uncertainties.sum(dim=0) / (probs.shape[0] - 1)
    mean_epistemic_uncertainties_1 = mean_epistemic_uncertainties_0.sum(dim=0) / (probs.shape[0])

    return mean_epistemic_uncertainties_1

def calc_pcs(
        probs,
):
    # Mean and standard deviation of probabilities
    if probs.shape[0] == 1:
        return probs[0]

    mean_probs = probs.mean(dim=0)
    std_probs = probs.std(dim=0)

    top_v = torch.max(mean_probs, 1-mean_probs)

    pcs = (2*top_v-1) / (2*std_probs+1e-10)

    constrained_pcs = 1-torch.exp(-pcs)

    rescaled_conf = calc_rescaled_conf(probs)

    return constrained_pcs*rescaled_conf

def calc_rescaled_conf(
        probs,
):
    mean_probs = probs.mean(dim=0)
    inv_mean_probs = 1 - mean_probs
    max_probs = torch.max(mean_probs, inv_mean_probs)
    assert (max_probs <= 1).all(), "Max probabilities exceed 1"
    assert (max_probs >= 0.5).all(), "Max probabilities less than 0.5"

    # recale to [0, 1]
    rescaled_conf = 2 * (max_probs - 0.5)
    return rescaled_conf

def calc_uncertainty(
        probs,
        mode,
):
    # Remove gradients from probs
    probs = probs.detach()
    mean_probs = probs.mean(dim=0)
    if mode == 'mean':
        rescaled_conf = calc_rescaled_conf(probs)
        return 1-rescaled_conf
    elif mode == 'predictive':
        predictive_uncertainty = calc_entropy(mean_probs)

        indices, values = torch.where(predictive_uncertainty > 1)
        for index, value in zip(indices, values):
            print(f"Predictive uncertainty at index {index} exceeds 1: {value.item()}")

        assert (predictive_uncertainty <= 1).all(), "Predictive uncertainty exceeds 1"

        return predictive_uncertainty
    elif mode == 'aleatoric':
        aleatoric_uncertainty = calc_entropy(probs)
        aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=0)

        assert (aleatoric_uncertainty <= 1).all(), "Aleatoric uncertainty exceeds 1"

        return aleatoric_uncertainty
    elif mode == 'epistemic':
        epistemic_uncertainty = calc_epistemic_uncertainty(probs)
        
        assert (epistemic_uncertainty <= 1).all(), "Epistemic uncertainty exceeds 1"
        assert (epistemic_uncertainty >= 0).all(), "Epistemic uncertainty is negative"

        return epistemic_uncertainty
    elif mode == 'epistemic_kl':
        epistemic_uncertainty = calc_epistemic_uncertainty(probs, metric='kl_div')

        return epistemic_uncertainty
    elif mode == "pcs":
        return 1-calc_pcs(probs)
    else:
        raise ValueError(f"Unknown uncertainty mode: {mode}. Choose from 'predictive', 'aleatoric', 'epistemic', or 'pcs'.")

def vectorized_loss(probs, labels, processed_weights, R, train_eval_index):
    n_heads, batch_size, num_classes = probs.shape

    labels_expanded = labels.unsqueeze(0).expand(n_heads, batch_size, num_classes)
    constr_output = get_constr_out_vectorized(probs, R)  # [N, batch_size, num_classes]
    train_output = labels_expanded * probs
    train_output = get_constr_out_vectorized(train_output, R)
    train_output = (1 - labels_expanded) * constr_output + labels_expanded * train_output

    losses = F.binary_cross_entropy(
        train_output[:, :, train_eval_index], 
        labels_expanded[:, :, train_eval_index], 
        reduction='none'
    )  # [N, batch_size, len(train_eval_index)]

    processed_weights_expanded = processed_weights.unsqueeze(0).expand(n_heads, -1, -1)

    losses = losses * processed_weights_expanded  # Broadcast as needed

    predictions = constr_output.detach()
    return losses, predictions