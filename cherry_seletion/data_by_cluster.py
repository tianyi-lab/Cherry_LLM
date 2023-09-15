import numpy as np
from sklearn.cluster import  KMeans
from sklearn.manifold import TSNE
import torch
import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, required=True)
    parser.add_argument("--json_data_path", type=str, required=True)
    parser.add_argument("--json_save_path", type=str, required=True)
    parser.add_argument("--sent_type", type=int, default=0)
    parser.add_argument("--ppl_type", type=int, default=0)
    parser.add_argument("--cluster_method", type=str, default='kmeans')
    parser.add_argument("--reduce_method", type=str, default='tsne')
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--kmeans_num_clusters", type=int, default=100)
    parser.add_argument("--low_th", type=int, default=1)
    parser.add_argument("--up_th", type=int, default=99)

    args = parser.parse_args()
    return args

def do_clustering(args, high_dim_vectors):

    clustering_algorithm = args.cluster_method
    if clustering_algorithm == 'kmeans':
        clustering = KMeans(n_clusters=args.kmeans_num_clusters, random_state=0).fit(high_dim_vectors)
    
    return clustering

def do_reduce_dim(args, high_dim_vectors):
    # Perform t-SNE for visualization
    if args.reduce_method == 'tsne':
        tsne = TSNE(n_components=2, random_state=0)
        low_dim_vectors = tsne.fit_transform(high_dim_vectors)
    return low_dim_vectors

def sample_middle_confidence_data(cluster_labels, confidences, n, low_th=25, up_th=75):
    num_clusters = len(np.unique(cluster_labels))

    # Get the indices for each cluster
    cluster_indices = {i: np.where(cluster_labels == i)[0] for i in range(num_clusters)}
    
    # Create a dictionary to store the indices of the middle level confidence samples
    middle_confidence_samples = {}

    for i in range(num_clusters):
        # Get the sorted indices for this cluster
        sorted_indices = cluster_indices[i]
        
        # If there are less than n samples in this class, just return all of them
        if len(sorted_indices) < n:
            middle_confidence_samples[i] = sorted_indices
            continue

        # Get the confidences for this cluster
        cluster_confidences = confidences[sorted_indices]
        lower_threshold = np.percentile(cluster_confidences, low_th)
        upper_threshold = np.percentile(cluster_confidences, up_th)

        # Get the indices of the samples within the middle level confidence range
        middle_indices = sorted_indices[(cluster_confidences >= lower_threshold) & (cluster_confidences <= upper_threshold)]
        
        # If there are less than n samples in the middle range, use all of them
        if len(middle_indices) < n:
            middle_confidence_samples[i] = middle_indices
        else:
            # Calculate step size for even sampling
            step_size = len(middle_indices) // n
            # Select evenly from the middle level confidence samples
            middle_confidence_samples[i] = middle_indices[::step_size][:n]

    return middle_confidence_samples

def main():

    args = parse_args()
    print(args)

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    emb_list = []
    ppl_list = []
    for i in tqdm(range(len(pt_data))):
        data_i = pt_data[i]
        sent_emb_list = data_i['sent_emb']
        emb_list.append(sent_emb_list[args.sent_type])
        ppl_list.append(data_i['ppl'][args.ppl_type].item())

    high_dim_vectors = torch.cat(emb_list,0).numpy()
    ppl_array = np.array(ppl_list)

    clustering = do_clustering(args, high_dim_vectors)
    cluster_labels = clustering.labels_

    def get_json_sample(middle_confidence_samples):
        json_samples = []
        for k in middle_confidence_samples.keys():
            ids_list = middle_confidence_samples[k].tolist()
            for id_i in ids_list:
                ori_sample = json_data[id_i]
                json_samples.append(ori_sample)

        return json_samples

    middle_confidence_samples = sample_middle_confidence_data(cluster_labels, ppl_array, args.sample_num, args.low_th, args.up_th)
    new_data = get_json_sample(middle_confidence_samples)
    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)
    pass


if __name__ == '__main__':
    main()