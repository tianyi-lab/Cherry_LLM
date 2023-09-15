python cherry_seletion/data_by_cluster.py \
    --pt_data_path alpaca_data_pre.pt \
    --json_data_path data/alpaca_data.json \
    --json_save_path alpaca_data_pre.json \
    --sample_num 10 \
    --kmeans_num_clusters 100 \
    --low_th 25 \
    --up_th 75