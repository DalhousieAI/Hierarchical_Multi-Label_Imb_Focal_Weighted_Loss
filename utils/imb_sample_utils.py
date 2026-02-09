import numpy as np
import networkx as nx
import math
import random
from collections import defaultdict
from datetime import datetime
from collections import deque
import os

def get_hierarchies(digraph):
    
    digraph = digraph.reverse()
    hierarchies = []

    def dfs(node, path):
        # Append the current node to the path
        current_path = path + [node]
        hierarchies.append(current_path)
        # Traverse neighbors
        for neighbor in digraph.successors(node):
            dfs(neighbor, current_path)

    root = 'root'
    # Perform DFS to collect all paths
    dfs(root, [])

    # Format hierarchies
    return ['.'.join(path[1:]) if len(path) > 1 else path[0] for path in hierarchies]

def process_arff_to_sample_counts(file_path):
    # Step 1: Extract last values from the ARFF file
    last_values = []
    with open(file_path, 'r') as file:
        in_data_section = False
        for line in file:
            line = line.strip()
            if line.lower() == "@data":
                in_data_section = True
                continue
            if in_data_section and line:
                last_values.append(line.split(",")[-1])
    
    # Step 2: Split elements with '@' and create a flattened list
    processed_list = []
    for item in last_values:
        if '@' in item:
            processed_list.extend(item.split('@'))
        else:
            processed_list.append(item)

    # Step 3: Replace '/' with '.' in all elements
    processed_list = [element.replace('/', '.') for element in processed_list]
    
    # Step 3: Count occurrences of each unique element
    element_counts = {}
    for element in processed_list:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1
    
    return element_counts

# Function to create the new dictionary
def get_term_count_dict_with_hierarchy(terms, counts):
    # Initialize the new dictionary
    new_dict = {}
    
    # Iterate over the terms
    for term in terms:
        # Get the leaf node (last part after splitting by '.')
        leaf = term.split('.')[-1]
        
        # Get the value from the counts dictionary or default to 0
        value = counts.get(leaf, 0)
        
        # Add to the new dictionary
        new_dict[term] = value
    
    return new_dict

def update_counts(existing_collection_of_counts_dict, new_sample_count_dict):
    # Make a copy of new_sample_count_dict to avoid modifying the original
    updated_new_counts = new_sample_count_dict.copy()

    # Sort keys by the depth of their hierarchy (i.e., number of dots in the key) in descending order
    sorted_keys = sorted(updated_new_counts.keys(), key=lambda x: x.count('.'), reverse=True)

    # Propagate counts up the hierarchy in new_sample_count_dict
    for key in sorted_keys:
        # Skip the 'root' key itself
        if key == 'root':
            continue

        # Find the parent of the current key
        if '.' in key:
            parent = key.rsplit('.', 1)[0]  # Parent is everything before the last '.'
            # Add the current key's value to its parent
            updated_new_counts[parent] = updated_new_counts.get(parent, 0) + updated_new_counts[key]
        else:
            # If no '.' in key, add the value to the 'root'
            updated_new_counts['root'] = updated_new_counts.get('root', 0) + updated_new_counts[key]
    updated_counts = existing_collection_of_counts_dict.copy()
    for key, value in updated_new_counts.items():
        updated_counts[key] = updated_counts.get(key, 0) + value

    return updated_counts

def get_dict_aggregated_count_without_leaf_nodes_with_zero_samples(
        all_terms_agg_count_dict_with_hierarchy
        ):
    countDict_agg = all_terms_agg_count_dict_with_hierarchy.copy()
    keys_to_remove = [key for key, value in countDict_agg.items() if value == 0]
    for key in keys_to_remove:
        del countDict_agg[key]
    return countDict_agg

def get_nodes_in_hierarchy_with_no_samples(all_terms_agg_count_dict_with_hierarchy):
    agg_count = all_terms_agg_count_dict_with_hierarchy.copy()
    zero_pairs = []
    for key, value in agg_count.items():
        if value == 0:
            zero_pairs.append(key)
    zero_pairs.sort(reverse=True)
    return zero_pairs

def get_g_graph_with_no_zero_samples(original_g, zero_pairs):
    dataset_graph = original_g.copy()
    for element in zero_pairs:
        if element in dataset_graph.nodes:
            dataset_graph.remove_node(element)
    return dataset_graph

def calculate_imbalance_ratios(all_terms_agg_count_dict_with_hierarchy):
    countDict = get_dict_aggregated_count_without_leaf_nodes_with_zero_samples(
        all_terms_agg_count_dict_with_hierarchy
        )
    countDict.pop('root', None)
    maxCount = max(value for key, value in countDict.items() if key != 'root')
    IRLbP = {}
    for path in countDict:
        pathCount = countDict[path]
        IRLbP[path] = maxCount / pathCount
    HMeanIR = sum(IRLbP[path] for path in countDict) / len(countDict)
    return IRLbP, HMeanIR

def retrieve_minority_paths(
        HMeanIR_original, 
        use_HMeanIR_original,
        all_terms_agg_count_dict_with_hierarchy,
        ):
    IRLbP, HMeanIR = calculate_imbalance_ratios(all_terms_agg_count_dict_with_hierarchy)
    if(use_HMeanIR_original):
        HMeanIR = HMeanIR_original
    minPaths = []
    countDict = get_dict_aggregated_count_without_leaf_nodes_with_zero_samples(
        all_terms_agg_count_dict_with_hierarchy
    )
    countDict.pop('root', None)
    for path in countDict:
        if IRLbP[path] > HMeanIR:
            minPaths.append(path)
    return minPaths, HMeanIR


def get_rows_from_children_with_hierarchy(parent_label_name, data):
    return list(filter(
    lambda row: any(
        label.startswith(parent_label_name)
        for label in row.split(',')[-1].split('@')
    ),
    data))
    

def get_rows_from_children_without_hierarchy(
        parent_label_name, 
        data,
        all_terms_agg_count_dict_with_hierarchy,
        ):
    
    child_label_names_list = []
    for key in all_terms_agg_count_dict_with_hierarchy.keys():
        if parent_label_name in key:
            # Split the key by '.' and find the index of the substring
            parts = key.split('.')
            if parent_label_name in parts:
                # Add everything after the substring to the child_label_name
                index = parts.index(parent_label_name)
                child_label_names_list.extend(parts[index + 1:])
    
    # Filter rows containing any of the labels
    matching_rows = list(filter(
        lambda row: any(label in row.split(',')[-1].split('@') for label in child_label_names_list),
        data
    ))

    return matching_rows

    
def update_train_with_duplicated_data(resampled_train, all_duplicated_data):
    
    R = np.zeros(resampled_train.A.shape)
    np.fill_diagonal(R, 1)
    ancestors_mapping_to_Y_values = {}
    for i in range(len(resampled_train.A)):
        ancestors = list(nx.descendants(resampled_train.g, list(resampled_train.g.nodes)[i])) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
        Y_list = [0] * len(resampled_train.terms)
        if ancestors:
            for element in ancestors:
                if element in resampled_train.terms:
                    # Find the index of the element in resampled_train.terms
                    index = resampled_train.terms.index(element)
                    # Set the corresponding index in Y_list to 1
                    Y_list[index] = 1
            
        for list(resampled_train.g.nodes)[i] in resampled_train.terms:
            index = resampled_train.terms.index(list(resampled_train.g.nodes)[i])
            Y_list[index] = 1

        # Fill the dictionary in a loop
        
        ancestors_mapping_to_Y_values[list(resampled_train.g.nodes)[i]] = Y_list  # Add key-value pair

    # Replace '.' with '/' in keys, modifying the dictionary in place
    for key in list(ancestors_mapping_to_Y_values.keys()):  # Use list() to avoid runtime modification error
        new_key = key.replace('.', '/')
        ancestors_mapping_to_Y_values[new_key] = ancestors_mapping_to_Y_values.pop(key)

    # Initialize two lists for the split
    duplicated_X_rows = []  # To store all values except the last one
    duplicated_Y_rows = []  # To store the last value

    # Split each string in the input list
    for element in all_duplicated_data:
        # Split the string by commas
        parts = element.split(",")
        
        # Add all except the last value to duplicated_X_rows as a list
        duplicated_X_rows.append(parts[:-1] if len(parts) > 1 else [])
        
        # Add the last value to duplicated_Y_rows as a single-element list
        duplicated_Y_rows.append([parts[-1]])

    # Append the new rows using np.vstack
    # Convert duplicated_X_rows to match the dtype of resampled_train.X
    duplicated_X_rows = np.array(duplicated_X_rows, dtype=resampled_train.X.dtype)
    resampled_train.X = np.vstack([resampled_train.X, duplicated_X_rows])

    # Step 1: Flatten the list of lists
    flattened_list = [item[0] for item in duplicated_Y_rows]

    # Step 2: Process each key
    result_list = []
    for key in flattened_list:
        # Split key at '@' and retrieve values
        sub_keys = key.split('@')
        value_lists = [ancestors_mapping_to_Y_values[sub_key] for sub_key in sub_keys]
        
        # Combine values (strictly 1 or 0)
        combined_result = [1 if any(value[i] for value in value_lists) else 0 for i in range(len(value_lists[0]))]
        result_list.append(combined_result)

    result_list = np.array(result_list, dtype=resampled_train.Y.dtype)
    resampled_train.Y = np.vstack([resampled_train.Y, result_list])

    return resampled_train

def duplicate_samples(
        data, 
        number_of_samples_to_duplicate, 
        label_name, 
        is_GO, 
        all_terms_agg_count_dict_with_hierarchy
        ):
    if not is_GO:
        label_name = label_name.replace('.', '/')
    # Find the rows containing the label_name
    matching_rows = [row for row in data if label_name in row.split(',')[-1].split('@')]
    
    if not matching_rows:
        if is_GO:
            matching_rows = get_rows_from_children_without_hierarchy(
                label_name, 
                data, 
                all_terms_agg_count_dict_with_hierarchy
                )
        else:
            matching_rows = get_rows_from_children_with_hierarchy(label_name, data)
    
    # Replace the last comma-separated value with "str" for matching rows
    modified_rows = [
        ','.join(row.split(',')[:-1] + [label_name])
        for row in matching_rows
    ]
    matching_rows = modified_rows
    # Randomly duplicate samples
    if not matching_rows:
        print(f"No rows found with the label: {label_name}")
        return [], {}
    duplicated_rows = random.choices(matching_rows, k=number_of_samples_to_duplicate)
    
    # Calculate counts per label in duplicated rows
    counts_dict = defaultdict(int)
    for row in duplicated_rows:
        if is_GO:
            labels = row.split(',')[-1].split('@')  # Extract labels from the row
        else:
            labels = row.split(',')[-1].replace('/', '.').split('@')  # Extract labels from the row
        for label in labels:
            counts_dict[label] += 1

    return duplicated_rows, dict(counts_dict)

def process_arff_file(input_file_path):
    # Read the ARFF file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    # Separate header and data
    header = []
    data = []
    in_data_section = False
    
    for line in lines:
        if line.strip().lower() == '@data':
            in_data_section = True
            header.append(line.strip())
        elif in_data_section:
            data.append(line.strip())
        else:
            header.append(line.strip())
    
    return header, data

def get_leaf_nodes_last_depth(labelTree_reverse):
    root_nodes = [node for node in labelTree_reverse.nodes if labelTree_reverse.in_degree(node) == 0]

    if not root_nodes:
        raise ValueError("No root nodes found. Ensure the graph is a DAG.")

    depths = {node: 0 for node in labelTree_reverse.nodes}  # Initialize all depths to 0
    queue = deque(root_nodes)  # Start from all root nodes

    while queue:
        node = queue.popleft()
        current_depth = depths[node]
        for neighbor in labelTree_reverse.successors(node):
            depths[neighbor] = max(depths[neighbor], current_depth + 1)
            queue.append(neighbor)

    max_depth = max(depths.values())

    leaf_nodes_last_depth = [
        node for node in labelTree_reverse.nodes
        if labelTree_reverse.out_degree(node) == 0 and depths[node] == max_depth
    ]

    return leaf_nodes_last_depth

def updated_transformed_data_from_train_X(train,data):
    # Replace the values
    for i in range(len(data)):
        parts = data[i].split(",")  # Split the string into parts
        replacement_values = ",".join(map(str, train.X[i]))  # Convert train_X[i] values to strings directly
        parts[:-1] = replacement_values.split(",")  # Replace all but the last with train_X[i] values
        data[i] = ",".join(parts)  # Reassemble the string
    return data

def hros_pd(
        dataset_graph, 
        train, 
        data, 
        all_hierarchies_list, 
        all_terms_agg_count_dict_with_hierarchy, 
        is_GO, 
        S=0.1
        ):
    countDict = get_dict_aggregated_count_without_leaf_nodes_with_zero_samples(
        all_terms_agg_count_dict_with_hierarchy
    )
    labelPaths = all_terms_agg_count_dict_with_hierarchy.copy()
    labelPaths.pop('root', None)
    samplesToCreate = int(countDict['root'] * S)
    labelTree = dataset_graph
    minPaths, HMeanIR = retrieve_minority_paths(
        None, 
        False,
        all_terms_agg_count_dict_with_hierarchy,
        )
    meanSize = math.ceil(countDict['root'] / (len(countDict)-1))
    maxIncrease = math.ceil(samplesToCreate / len(minPaths))
    resampled_train = train
    labelTree_reverse = labelTree.reverse()
    all_duplicated_data = []
    while labelTree_reverse.number_of_nodes() > 1:
        leafNodes = get_leaf_nodes_last_depth(labelTree_reverse)
        if len(leafNodes) == 0:
            break
        countDict = \
            get_dict_aggregated_count_without_leaf_nodes_with_zero_samples(
                all_terms_agg_count_dict_with_hierarchy
                )
        for leafNode in leafNodes:
            if is_GO:
                paths_with_leaf_node = [item for item in minPaths if item.split('.')[-1] == train.terms[leafNode]]
            else:
                paths_with_leaf_node = [item for item in minPaths if item == train.terms[leafNode]]
            if paths_with_leaf_node: # this list could contain one or more paths with same leaf node
                numSamples = countDict[paths_with_leaf_node[0]]
                number_of_samples_to_duplicate = min(maxIncrease, meanSize - numSamples)
                if number_of_samples_to_duplicate > 0:
                    if is_GO:
                        duplicated_sample_rows, duplicated_sample_counts_without_hierarchy = duplicate_samples(
                            data, 
                            number_of_samples_to_duplicate, 
                            train.terms[leafNode], 
                            True, 
                            all_terms_agg_count_dict_with_hierarchy
                            ) #,all_hierarchies_list[paths_with_leaf_node[0]])
                    else:
                        duplicated_sample_rows, duplicated_sample_counts_without_hierarchy = duplicate_samples(
                            data, 
                            number_of_samples_to_duplicate, 
                            train.terms[leafNode], 
                            False,
                            all_terms_agg_count_dict_with_hierarchy
                            ) #,all_hierarchies_list[paths_with_leaf_node[0]])
                    all_duplicated_data += duplicated_sample_rows
                    total_counts = defaultdict(int)
                    for label, count in duplicated_sample_counts_without_hierarchy.items():
                        total_counts[label] += count
                    if is_GO:
                        label_count_new_samples_without_hierarchy = dict(total_counts)
                        label_count_new_samples_with_hierarchy=get_term_count_dict_with_hierarchy(all_hierarchies_list,label_count_new_samples_without_hierarchy)
                    else:
                        label_count_new_samples_with_hierarchy = dict(total_counts)
                    all_terms_agg_count_dict_with_hierarchy = update_counts(all_terms_agg_count_dict_with_hierarchy, label_count_new_samples_with_hierarchy)
            labelTree_reverse.remove_node(leafNode)
        minPaths, _ = retrieve_minority_paths(
            HMeanIR, 
            True,
            all_terms_agg_count_dict_with_hierarchy,
            ) # Check if labelTree needs to be passed or labelTree_reverse here
    
    resampled_train = update_train_with_duplicated_data(resampled_train, all_duplicated_data)
    return resampled_train

def oversampled_pd(g, train, arff_file_path, is_GO, S=0.1):
    source_sample_count_dict = process_arff_to_sample_counts(arff_file_path)
    if is_GO:
        all_hierarchies_list = get_hierarchies(train.g)
        source_data_sample_count_dict_without_hierarchy = source_sample_count_dict
        source_sample_count_dict_with_hierarchy = get_term_count_dict_with_hierarchy(all_hierarchies_list,source_data_sample_count_dict_without_hierarchy)
    else:
        all_hierarchies_list = train.terms
        source_sample_count_dict_with_hierarchy = source_sample_count_dict
        init_merged_dict = {key: 0 for key in all_hierarchies_list}
        merged_dict = {key: init_merged_dict.get(key, 0) + source_sample_count_dict_with_hierarchy.get(key, 0) for key in set(init_merged_dict) | set(source_sample_count_dict_with_hierarchy)}
        source_sample_count_dict_with_hierarchy = merged_dict
    all_terms_agg_count_dict_with_hierarchy = dict.fromkeys(source_sample_count_dict_with_hierarchy.keys(), 0)
    all_terms_agg_count_dict_with_hierarchy = update_counts(all_terms_agg_count_dict_with_hierarchy, source_sample_count_dict_with_hierarchy)
    _, data = process_arff_file(arff_file_path)
    updated_transformed_data = updated_transformed_data_from_train_X(train,data)
    nodes_with_zero_samples = get_nodes_in_hierarchy_with_no_samples(all_terms_agg_count_dict_with_hierarchy)
    dataset_graph = get_g_graph_with_no_zero_samples(g, nodes_with_zero_samples)
    resampled_train_object = hros_pd(
        dataset_graph=dataset_graph, 
        train=train, 
        data=updated_transformed_data, 
        all_hierarchies_list=all_hierarchies_list,
        all_terms_agg_count_dict_with_hierarchy=all_terms_agg_count_dict_with_hierarchy,
        is_GO=is_GO, 
        S=S
        )
    return resampled_train_object