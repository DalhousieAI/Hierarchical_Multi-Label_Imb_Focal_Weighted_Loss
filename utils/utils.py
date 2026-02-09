"""
This code was adapted from https://github.com/lucamasera/AWX;
Obtained from https://github.com/EGiunchiglia/C-HMCNN/blob/master/utils/parser.py
"""

import argparse
import gc
import numpy as np
import networkx as nx
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constrained_ff import ConstrainedFFNNModel, get_constr_out

from itertools import chain

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import average_precision_score

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Dictionaries with number of features and number of labels for each dataset
input_dims = {'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'seq':529, 'spo':86}
output_dims_FUN = {'cellcycle':499, 'derisi':499, 'eisen':461, 'expr':499, 'gasch1':499, 'gasch2':499, 'seq':499, 'spo':499}
output_dims_GO = {'cellcycle':4122, 'derisi':4116, 'eisen':3570, 'expr':4128, 'gasch1':4122, 'gasch2':4128, 'seq':4130, 'spo':4116}
output_dims_others = {'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
output_dims = {'FUN':output_dims_FUN, 'GO':output_dims_GO, 'others':output_dims_others}

#Dictionaries with the hyperparameters associated to each dataset
hidden_dims_FUN = {'cellcycle':500, 'derisi':500, 'eisen':500, 'expr':1250, 'gasch1':1000, 'gasch2':500, 'seq':2000, 'spo':250}
hidden_dims_GO = {'cellcycle':1000, 'derisi':500, 'eisen':500, 'expr':4000, 'gasch1':500, 'gasch2':500, 'seq':9000, 'spo':500}
hidden_dims_others = {'diatoms':2000, 'enron':1000,'imclef07a':1000, 'imclef07d':1000}
hidden_dims = {'FUN':hidden_dims_FUN, 'GO':hidden_dims_GO, 'others':hidden_dims_others}
lrs_FUN = {'cellcycle':1e-4, 'derisi':1e-4, 'eisen':1e-4, 'expr':1e-4, 'gasch1':1e-4, 'gasch2':1e-4, 'seq':1e-4, 'spo':1e-4}
lrs_GO = {'cellcycle':1e-4, 'derisi':1e-4, 'eisen':1e-4, 'expr':1e-4, 'gasch1':1e-4, 'gasch2':1e-4, 'seq':1e-4, 'spo':1e-4}
lrs_others = {'diatoms':1e-5, 'enron':1e-5,'imclef07a':1e-5, 'imclef07d':1e-5}
lrs = {'FUN':lrs_FUN, 'GO':lrs_GO, 'others':lrs_others}
epochss_FUN = {'cellcycle':106, 'derisi':67, 'eisen':110, 'expr':20, 'gasch1':42, 'gasch2':123, 'seq':13, 'spo':115}
epochss_GO = {'cellcycle':62, 'derisi':91, 'eisen':123, 'expr':70, 'gasch1':122, 'gasch2':177, 'seq':45, 'spo':103}
epochss_others = {'diatoms':474, 'enron':133,'imclef07a':592, 'imclef07d':588}
epochss = {'FUN':epochss_FUN, 'GO':epochss_GO, 'others':epochss_others}

# HML datasets parsing functions
# Skip the root nodes 
to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150']

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class arff_data():
    def __init__(self, arff_file, is_GO, is_test=False):
        self.X, self.Y, self.A, self.terms, self.g = parse_arff(arff_file=arff_file, is_GO=is_GO, is_test=is_test)
        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i,j] = m[j]

            
def parse_arff(arff_file, is_GO=False, is_test=False):
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        for num_line, l in enumerate(f):
            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        if is_GO:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms)==1:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i-1]))
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()
                else:
                    _, f_name, f_type = l.split()
                    
                    if f_type == 'numeric' or f_type == 'NUMERIC':
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(lambda x,i: [float(x)] if x != '?' else [np.nan])
                        
                    else:
                        cats = f_type[1:-1].split(',')
                        cats_lens.append(len(cats))
                        d.append({key:to_categorical(i, len(cats)).tolist() for i,key in enumerate(cats)})
                        feature_types.append(lambda x,i: d[i].get(x, [0.0]*cats_lens[i]))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()
                
                X.append(list(chain(*[feature_types[i](x,i) for i, x in enumerate(d_line[:len(feature_types)])])))
                
                for t in lab.split('@'): 
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] = 1
                    y_[nodes_idx[t.replace('/', '.')]] = 1
                Y.append(y_)
        X = np.array(X)
        Y = np.stack(Y)

    return X, Y, nx.to_numpy_array(g, nodelist=nodes), nodes, g

def initialize_dataset(name, datasets):
    is_GO, train, val, test = datasets[name]
    return arff_data(train, is_GO), arff_data(val, is_GO), arff_data(test, is_GO, True)

def initialize_other_dataset(name, datasets):
    is_GO, train, test = datasets[name]
    return arff_data(train, is_GO), arff_data(test, is_GO, True)

def gen_parser_args_focal():
    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')

    # Required  parameter
    parser.add_argument('--dataset', type=str, default="cellcycle_GO", required=True,
                        help='dataset name, must end with: "_GO", "_FUN", or "_others"' )
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--device', type=int, default='0',
                        help='GPU (default:0)')
    parser.add_argument('--focal_k', type=int, default=2,
                        help='focal k parameter (default: 2)')
    parser.add_argument('--focal_min', type=float, default=0.25,
                        help='focal minimum value (default: 0.25)')
    parser.add_argument('--focal_mode', type=str, default='epistemic',
                        help='focal mode, can be "aleatoric", "epistemic", "predictive", or "pcs" (default: epistemic)')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='number of heads for multi-head model (default: 2)')
    parser.add_argument('--use_node_weighting', type=bool, default=False,
                        help='Whether to use node weighting (default: False)')
    args = parser.parse_args()

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

    return args

def gen_parser_args():
    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')

    # Required  parameter
    parser.add_argument('--dataset', type=str, default="cellcycle_GO", required=True,
                        help='dataset name, must end with: "_GO", "_FUN", or "_others"' )
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--device', type=int, default='0',
                        help='GPU (default:0)')
    parser.add_argument('--resample_train', type=bool, default=False,
                        help='resample the training set for HML imbalance (default: False)')
    parser.add_argument('--node_weighting', type=str, default='class-wise',
                        help='weighting method for nodes, \
                        can be "false" (str), "binary", or "class-wise" (default: class-wise)')
    parser.add_argument('--min_weight_const', type=float, default=0.25,
                        help='minimum weight constant (omega) (default: 0.25)')
    parser.add_argument('--omega_scheduler', type=str, default="none",
                        help='weighting scheduler, \
                        can be "none" (str), "lin", "exp", or "alt" (default: alt)')
    parser.add_argument('--mixed_loss_lambda', type=float, default=-1.,
                        help='lambda for mixed loss, \
                        can be -1 for none or float (default: None)')
    parser.add_argument('--k', type=int, default=3,
                        help='exponential scheduler power (default: 3)')
    args = parser.parse_args()

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

    if args.mixed_loss_lambda == -1:
        args.mixed_loss_lambda = None

    if args.node_weighting == "false":
        args.node_weighting = False

    return args

def set_seed(args, input_args=True):
    # Set seed
    if input_args:
        seed = args.seed
    else:
        seed = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(args):
    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]

    return data, ontology

def set_hyperparams(data, ontology):
    # Set the hyperparameters 
    batch_size = 4
    num_layers = 3
    dropout = 0.7
    non_lin = 'relu'
    hidden_dim = hidden_dims[ontology][data]
    lr = lrs[ontology][data]
    weight_decay = 1e-5
    num_epochs = epochss[ontology][data]
    hyperparams = {'batch_size':batch_size, 'num_layers':num_layers, 'dropout':dropout, 'non_lin':non_lin, 'hidden_dim':hidden_dim, 'lr':lr, 'weight_decay':weight_decay}
    return hyperparams, num_epochs

def load_dataset(args, datasets):
    dataset_name = args.dataset
    # Load the datasets
    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8),  torch.tensor(test.to_eval, dtype=torch.uint8)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)
    
    different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = np.uint8), dtype=torch.uint8)

    if val:
        return train, val, test, different_from_0
    else:
        return train, test, different_from_0
    
def compute_ancestor_matrix(train, device):
    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(train.A) # train.A is the matrix where the direct connections are stored 
    for i in range(len(train.A)):
        ancestors = list(nx.descendants(g, i)) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    #Transpose to get the descendants for each node 
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    root_nodes = [n for n,d in g.out_degree() if d==0] 

    return R, root_nodes, g

def rescale_input_data(args, train, test, device, val=None):
    # Rescale data and impute missing data
    if ('others' in args.dataset):
        scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X.astype(float)))
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
    train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        
    test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(device), torch.tensor(test.Y).to(device)

    if val:
        return train, val, test
    return train, test

def create_dataloaders(args, train, test, batch_size, val=None):
    #Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' not in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
        for (x, y) in zip(val.X, val.Y):
            train_dataset.append((x,y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    return train_loader, test_loader

def skip_root_eval(dataset_name):
    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 
    return num_to_skip

def create_model_optimizer_criterion(data, ontology, num_to_skip, hyperparams, R, device):
    hidden_dim = hyperparams['hidden_dim']
    lr = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']
    # Create the model
    model = ConstrainedFFNNModel(
        input_dim=input_dims[data], 
        hidden_dim=hidden_dim, 
        output_dim=output_dims[ontology][data]+num_to_skip, 
        hyperparams=hyperparams,
        R=R
    )

    model.to(device)
    print("Model on gpu", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer

def train_model(
        num_epochs, 
        model, 
        train_loader, 
        test_loader, 
        device, 
        optimizer, 
        R, 
        train, 
        test, 
        dataset_name, 
        seed, 
        imb_method, 
        weights=None,
        target_omega=1,
        scheduler="none",
        k=3,
        mixed_loss_lambda=0.5
        ):
    
    # Train Loop
    train_eval_index = train.to_eval.bool()

    # Intialize weighting mechanic
    min_weight = 1
    if weights is not None:
        original_weights_pos = weights[0].copy()
        original_weights_neg = weights[1].copy()

    if weights is not None and target_omega:
        min_weight = torch.Tensor(weights).min().item()
        if scheduler == "linear" or scheduler == "lin":
            delta_weights_pos = (target_omega - weights[0]) / (len(train_loader)-1)
            delta_weights_neg = (target_omega - weights[1]) / (len(train_loader)-1)
        elif scheduler == "exponential" or scheduler == "exp":
            assert k is not None
            delta_weights_pos = (target_omega - weights[0]) / (len(train_loader)-1)**k
            delta_weights_neg = (target_omega - weights[1]) / (len(train_loader)-1)**k
            
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
            
        for i, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = model(x.float())

            if target_omega:
                if scheduler == "linear" or scheduler == "lin":
                    weights[0] = original_weights_pos + delta_weights_pos*i
                    weights[1] = original_weights_neg + delta_weights_neg*i
                elif scheduler == "exponential" or scheduler == "exp":
                    weights[0] = original_weights_pos + delta_weights_pos*i**k
                    weights[1] = original_weights_neg + delta_weights_neg*i**k
                elif scheduler == "alternating" or scheduler == "alt":
                    if i % 2 == 0:
                        weights[0] = original_weights_pos
                        weights[1] = original_weights_neg
                    else:
                        weights[0] = target_omega * np.ones_like(weights[0])
                        weights[1] = target_omega * np.ones_like(weights[1])
                elif scheduler =="none":
                    pass
                else:
                    raise ValueError("Invalid scheduler. Only 'linear', 'exponential', 'step', and 'none' are supported.")

            #MCLoss
            constr_output = get_constr_out(output, R)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1-labels)*constr_output.double() + labels*train_output

            loss = F.binary_cross_entropy(train_output[:, train_eval_index], labels[:, train_eval_index], reduction='none')

            if mixed_loss_lambda is not None:
                unweighted_loss = F.binary_cross_entropy(train_output[:, train_eval_index], labels[:, train_eval_index])

            if weights is not None:
                applied_weights = torch.Tensor(weights).to(device).T[train_eval_index]

                # Turn weights from (n_nodes, 2) to (batch_size, n_nodes, 2)
                applied_weights = applied_weights.unsqueeze(0).expand(loss.size(0), -1, -1)

                # Use targets as mask for applied weights such that the end result is a (batch_size, n_nodes) tensor
                weight_mask = labels[:, train_eval_index].int()

                rows = torch.arange(applied_weights.shape[0], device=applied_weights.device).view(-1, 1).expand(-1, applied_weights.shape[1])
                cols = torch.arange(applied_weights.shape[1], device=applied_weights.device).view(1, -1).expand(applied_weights.shape[0], -1)

                applied_weights = applied_weights[rows, cols, weight_mask]

                loss = loss * (applied_weights.expand_as(loss).to(device))

            loss = loss.mean()

            if mixed_loss_lambda is not None:
                loss = mixed_loss_lambda*loss + (1-mixed_loss_lambda)*unweighted_loss

            predicted = constr_output.data > 0.5

            # Total number of labels
            total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train = (predicted == labels.byte()).sum()

            loss.backward()
            optimizer.step()

            # Loss and metric tracking
            epoch_loss += loss.item()
        
        # Print the loss and accuracy for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {correct_train/total_train:.4f}')


    del optimizer
    del train_loader
    del x, labels, output, loss, predicted
    del train_output, constr_output
    if weights is not None:
        del applied_weights, weight_mask, rows, cols
    if mixed_loss_lambda is not None:
        del unweighted_loss
    torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    # Test Loop
    model.eval()

    test_eval_index = test.to_eval.bool()

    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader):
                    
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float())
            predicted = constrained_output.data > 0.5
    
            #Move output and label back to cpu to be processed by sklearn
            predicted = predicted.to('cpu')
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = y
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, y), dim =0)

        optimistic_ap_score = average_precision_score(y_test[:, test_eval_index], constr_test.data[:, test_eval_index], average='micro')
        ap_score = average_precision_score(y_test[:, test_eval_index], predicted_test.data[:, test_eval_index], average='micro')
        f1, precision, recall, _, _, _ = node_f1(y_test[:, test_eval_index], predicted_test.data[:, test_eval_index])
        
        mode = scheduler

        if scheduler == "none" and mixed_loss_lambda is not None:
            mode = "mixed_loss"
        elif scheduler == "exponential":
            mode = mode + "_k" + str(k)

        if mixed_loss_lambda is None:
            mixed_loss_lambda_str = "none"
        else:
            mixed_loss_lambda_str = str(mixed_loss_lambda)
        f = open('results/' + dataset_name + '.csv', 'a')

        # If the file is empty, write the header
        if os.stat('results/' + dataset_name + '.csv').st_size == 0:
            f.write('imb_method,mode,seed,' + \
                    'target_omega,mixed_loss_lambda,' + \
                    'epoch,optimistic_ap_score,ap_score,' + \
                    'f1,precision,recall\n')

        f.write(imb_method +',' + mode + ',' + str(seed) + ',' \
                + str(target_omega) + ',' + mixed_loss_lambda_str + ',' \
                + str(epoch) + ',' + str(optimistic_ap_score) + ',' + str(ap_score) + ',' \
                + str(f1) + ',' + str(precision) + ',' + str(recall) +'\n')
        
        f.close()

        # Save constr_test.data to file
        pred_file_name = 'predictions/' + \
                dataset_name + '_' + imb_method + \
                    '_m-' + mode + '_mlamb-' + mixed_loss_lambda_str + \
                    '_w-' + str(min_weight) + '_s-' + str(seed)
        # replace "." with "-" in the file name
        pred_file_name = pred_file_name.replace(".", "-")
        pred_file_name = pred_file_name + '.pt'
        torch.save(constr_test.data, pred_file_name)
   
def obtain_node_frequencies(dataset, R=None):
    # Obtain the frequency of each node in the training set
    node_frequencies = dataset.Y.sum(0).to('cpu')
    inverse_node_frequencies = len(dataset.Y) - node_frequencies
    
    assert torch.all(node_frequencies >= 0) and torch.all(inverse_node_frequencies >= 0)

    if R is not None:
        R = R.to('cpu')
        for i in range(len(node_frequencies)):
            descendent_idxs = np.where(R[0][i] == 1)[0]
            node_frequencies[i] = node_frequencies[descendent_idxs].sum()
            inverse_node_frequencies[i] = inverse_node_frequencies[descendent_idxs].sum()
    
    # Ensure all elements in both frequencies are positive
    assert torch.all(node_frequencies >= 0) and torch.all(inverse_node_frequencies >= 0)

    dataset_len = len(dataset.Y)

    return node_frequencies, inverse_node_frequencies, dataset_len

def get_balanced_weights(
        node_frequencies, 
        inverse_node_frequencies, 
        dataset_len, 
        mode="class-wise", 
        no_positive=False, 
        no_negative=True,
        min_weight_const=0.25,
        common_weights=False
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

        node_weights = node_weights.numpy()

        return node_weights

    positive_node_weights = calc_balanced_weights(node_frequencies, total_classes, dataset_len)
    # Scale positive weights to be between 0 and 1
    min_pos_weight = positive_node_weights.min()
    max_pos_weight = positive_node_weights.max()

    positive_node_weights = min_weight_const + max_pos_weight * (positive_node_weights - min_pos_weight) / (max_pos_weight - min_pos_weight)

    negative_node_weights = calc_balanced_weights(inverse_node_frequencies, total_classes, dataset_len)
    # Scale negative weights to be between 0 and 1
    min_neg_weight = negative_node_weights.min()
    max_neg_weight = negative_node_weights.max()

    negative_node_weights = min_weight_const + max_neg_weight * (negative_node_weights - min_neg_weight) / (max_neg_weight - min_neg_weight)
    
    if common_weights:
        negative_node_weights = positive_node_weights

    if no_positive:
        positive_node_weights = np.ones_like(positive_node_weights)
    if no_negative:
        negative_node_weights = np.ones_like(negative_node_weights)

    return positive_node_weights, negative_node_weights


# Metric utils
def node_f1(labels, predictions):
    node_tp_dict = {}
    node_fp_dict = {}
    node_fn_dict = {}
    node_precision_dict = {}
    node_recall_dict = {}
    node_f1_dict = {}

    labels = labels.T
    predictions = predictions.T

    # Get the true positives, false positives, and false negatives for each node
    for node in range(len(labels)):
        label_at_node = labels[node]
        pred_at_node = predictions[node]

        node_tp_dict[node] = (label_at_node == 1) & (pred_at_node == 1)
        node_fp_dict[node] = (label_at_node == 0) & (pred_at_node == 1)
        node_fn_dict[node] = (label_at_node == 1) & (pred_at_node == 0)

        node_tp_count = node_tp_dict[node].sum()
        node_fp_count = node_fp_dict[node].sum()
        node_fn_count = node_fn_dict[node].sum()
        
        if node_tp_count + node_fp_count == 0:
            node_precision_dict[node] = 0
        else:
            node_precision_dict[node] = (node_tp_count / (node_tp_count + node_fp_count)).item()
        
        if node_tp_count + node_fn_count == 0:
            node_recall_dict[node] = 0
        else:
            node_recall_dict[node] = (node_tp_count / (node_tp_count + node_fn_count)).item()

        if node_precision_dict[node] + node_recall_dict[node] == 0:
            node_f1_dict[node] = 0
        else:
            node_f1_dict[node] = 2 * (node_precision_dict[node] * node_recall_dict[node]) / (node_precision_dict[node] + node_recall_dict[node])
    
    macro_average_f1 = sum(node_f1_dict.values()) / len(node_f1_dict)
    macro_average_precision = sum(node_precision_dict.values()) / len(node_precision_dict)
    macro_average_recall = sum(node_recall_dict.values()) / len(node_recall_dict)

    return macro_average_f1, macro_average_precision, macro_average_recall, node_f1_dict, node_precision_dict, node_recall_dict