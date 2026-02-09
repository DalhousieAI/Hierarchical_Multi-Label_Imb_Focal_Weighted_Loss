import copy

import numpy as np

from utils.notebook_args import NotebookArgs
from utils.utils import set_seed, get_dataset, \
    set_hyperparams, load_dataset, compute_ancestor_matrix, \
    rescale_input_data, create_dataloaders, skip_root_eval, \
    create_model_optimizer_criterion, train_model, \
    obtain_node_frequencies, get_balanced_weights, \
    gen_parser_args
from utils.imb_sample_utils import oversampled_pd
from utils.datasets import DATASETS

if __name__ == "__main__":
    args = gen_parser_args()

    dataset_name = args.dataset
    data_hierarchy = dataset_name.split("_")[1]
    is_GO = data_hierarchy == "GO"

    resample_train = args.resample_train
    node_weighting = args.node_weighting
    min_weight_const = args.min_weight_const
    scheduler = args.omega_scheduler
    mixed_loss_lambda = args.mixed_loss_lambda
    k = args.k

    train_dataset_path = f"./HMC_data/datasets_{data_hierarchy}/{dataset_name}/{dataset_name}.train.arff"

    using_node_weighting = node_weighting is not False
    if resample_train or using_node_weighting:
        assert using_node_weighting != resample_train, "Node_weighting and resample_train cannot be both True"

    if resample_train:
        imb_method = "hml_resample"
    elif using_node_weighting:
        imb_method = "node_weighting"
    else:
        imb_method = "none"
    
    # Print the arguments
    print(args)

    set_seed(args)
    data, ontology = get_dataset(args)
    hyperparams, num_epochs = set_hyperparams(data=data, ontology=ontology)

    contains_val = False

    datasets_output = load_dataset(args=args, datasets=DATASETS)

    if len(datasets_output) == 3:
        train, test, different_from_0 = datasets_output
    elif len(datasets_output) == 4:
        train, val, test, different_from_0 = datasets_output
        contains_val = True
    else:
        raise ValueError("The datasets_output should have 3 or 4 elements")

    R, root_nodes, g = compute_ancestor_matrix(train=train, device=args.device)

    assert len(root_nodes) == 1, "The dataset appears to contain more than one root node. Default over/under-sampling is not supported for this case."

    root_node = root_nodes[0]

    print("Root node:", root_node)

    original_train = None

    if resample_train:
        original_train = copy.deepcopy(train)
        original_val = copy.deepcopy(val) if contains_val else None
        original_test = copy.deepcopy(test)
        train = oversampled_pd(g=g, train=train, arff_file_path=train_dataset_path, is_GO=is_GO)
    if contains_val:
        train, val, test = rescale_input_data(args=args, train=train, val=val, test=test, device=args.device)
        if original_train is not None:
            original_train, _, _ = rescale_input_data(args=args, train=original_train, val=original_val, test=original_test, device=args.device)
        train_loader, test_loader = create_dataloaders(args=args, train=train, val=val, test=test, batch_size=hyperparams["batch_size"])
    else:
        train, test = rescale_input_data(args=args, train=train, test=original_test, device=args.device)
        if original_train is not None:
            original_train, _, _ = rescale_input_data(args=args, train=original_train, test=test, device=args.device)
        train_loader, test_loader = create_dataloaders(args=args, train=train, test=test, batch_size=hyperparams["batch_size"])

    num_to_skip = skip_root_eval(dataset_name=args.dataset)

    model, optimizer = create_model_optimizer_criterion(data=data, ontology=ontology, num_to_skip=num_to_skip, hyperparams=hyperparams, R=R, device=args.device)

    if original_train is not None:
        node_frequencies, inverse_node_frequencies, dataset_len = obtain_node_frequencies(dataset=original_train, R=R)
    else:
        node_frequencies, inverse_node_frequencies, dataset_len = obtain_node_frequencies(dataset=train, R=R)

    if node_weighting:
        positive_node_weights, negative_node_weights = get_balanced_weights(
            node_frequencies=node_frequencies, 
            inverse_node_frequencies=inverse_node_frequencies, 
            dataset_len=dataset_len,
            mode=node_weighting,
            min_weight_const=min_weight_const
            )
        node_weights = np.array([positive_node_weights, negative_node_weights])
        imb_method = imb_method + "_" + node_weighting
    else:
        node_weights = None

    # Results saved with columns: imb_method, seed, number of training epochs, optimistic ap score, ap score
    train_model(
        num_epochs=num_epochs, 
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        device=args.device, 
        optimizer=optimizer, 
        R=R, 
        train=train,
        test=test, 
        dataset_name=args.dataset, 
        seed=args.seed,
        imb_method=imb_method,
        weights=node_weights,
        target_omega=min_weight_const,
        scheduler=scheduler,
        k=k,
        mixed_loss_lambda=mixed_loss_lambda,
        )