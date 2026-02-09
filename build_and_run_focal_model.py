import copy
import numpy as np
import torch

from utils.notebook_args import NotebookArgs
from utils.utils import set_seed, get_dataset, set_hyperparams, \
    load_dataset, compute_ancestor_matrix, rescale_input_data, \
    create_dataloaders, skip_root_eval, obtain_node_frequencies, \
    get_balanced_weights, gen_parser_args_focal
from utils.datasets import DATASETS
from utils.imb_sample_utils import oversampled_pd
from utils.conf_hml_loss import ConfHMLLossWithProbs
from utils.multihead_utils import build_multi_headed_model

if __name__ == "__main__":
    args = gen_parser_args_focal()

    dataset_name = args.dataset
    data_hierarchy = dataset_name.split("_")[1]
    is_GO = data_hierarchy == "GO"
    seed = args.seed
    device= args.device
    resample_train = False
    node_weighting = "class-wise" # False, "binary", or "class-wise"
    min_weight_const = 0.25 # None or float
    scheduler = "none" # "none", "lin", "exp", or "alt"
    mixed_loss_lambda = None # can be None if not using mixed loss
    k = 1 # 3 is default if not passed to train model function - only applied if scheduler is exp

    # Multihead parameters
    focal_k = args.focal_k
    focal_min = args.focal_min
    focal_mode = args.focal_mode # "aleatoric, "epistemic", "predictive", or "pcs"
    n_heads = args.n_heads

    using_node_weighting = node_weighting is not False
    if resample_train or using_node_weighting:
        assert using_node_weighting != resample_train, "Node_weighting and resample_train cannot be both True"

    if resample_train:
        imb_method = "hml_resample"
    elif using_node_weighting:
        imb_method = "node_weighting"
    else:
        imb_method = "none"

    train_dataset_path = f"./HMC_data/datasets_{data_hierarchy}/{dataset_name}/{dataset_name}.train.arff"

    set_seed(args)
    data, ontology = get_dataset(args)
    hyperparams, num_epochs = set_hyperparams(data=data, ontology=ontology)

    print(args)
    print(hyperparams)
    print("num_epochs:", num_epochs)

    contains_val = False

    datasets_output = load_dataset(args=args, datasets=DATASETS)

    if len(datasets_output) == 3:
        train, test, different_from_0 = datasets_output
    elif len(datasets_output) == 4:
        train, val, test, different_from_0 = datasets_output
        contains_val = True
    else:
        raise ValueError("The datasets_output should have 3 or 4 elements")

    R, root_nodes, g = compute_ancestor_matrix(train=train, device=device)

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
        train, val, test = rescale_input_data(args=args, train=train, val=val, test=test, device=device)
        if original_train is not None:
            original_train, _, _ = rescale_input_data(args=args, train=original_train, val=original_val, test=original_test, device=device)
        train_loader, test_loader = create_dataloaders(args=args, train=train, val=val, test=test, batch_size=hyperparams["batch_size"])
    else:
        train, test = rescale_input_data(args=args, train=train, test=original_test, device=device)
        if original_train is not None:
            original_train, _, _ = rescale_input_data(args=args, train=original_train, test=test, device=device)
        train_loader, test_loader = create_dataloaders(args=args, train=train, test=test, batch_size=hyperparams["batch_size"])

    print("Length of training data:", len(train.X))

    num_to_skip = skip_root_eval(dataset_name=dataset_name)

    train_eval_index=train.to_eval.bool()
    test_eval_index=test.to_eval.bool()

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
        node_weights = np.asarray([positive_node_weights, negative_node_weights])
        imb_method = imb_method + "_" + node_weighting
    else:
        node_weights = None

    data_key = dataset_name.split("_")[0]

    model = build_multi_headed_model(
            n_heads=n_heads,
            data_str=data_key,
            hyperparams=hyperparams,
            ontology=ontology,
            num_to_skip=num_to_skip,
            backbone=None,
            backbone_checkpoint_path=None,
            R=R,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"]
    )

    conf_hml_loss = ConfHMLLossWithProbs(
        k=focal_k,
        R=R, 
        weights=node_weights,
        focal_min=focal_min,
        train_eval_index=train_eval_index,
        mode=focal_mode,
        use_node_weighting=args.use_node_weighting,
    )

    conf_hml_loss = conf_hml_loss.to(device)

    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        mean_loss, mean_acc, mean_unc, ap, bin_ap, f1, precision, recall = model.fit(
            train_loader=train_loader,
            optimizer=optimizer,
            conf_hml_loss=conf_hml_loss,
            device=device,
        )

        print(f"\tMean loss: {mean_loss}, Mean accuracy: {mean_acc}, Mean confidence: {1-mean_unc}")
        print(f"\tMean optimistic AP: {ap}, Mean bin AP: {bin_ap}")
        print(f"\tMean F1: {f1}, Mean precision: {precision}, Mean recall: {recall}")
    
    model.test(
        eval_loader=test_loader,
        test_eval_index=test_eval_index,
        conf_hml_loss=conf_hml_loss,
        seed=seed,
        num_epochs=num_epochs,
        num_heads=n_heads,
        focal_k=focal_k,
        focal_min=focal_min,
        dataset_name=dataset_name,
        conf_method=focal_mode,
        write_to_file=True,
        mix_enc_noise=1.0,
        device=device
    )