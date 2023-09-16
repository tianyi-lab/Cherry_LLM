array=(
    vicuna
    koala
    wizardlm
    sinstruct
    lima
)
for i in "${array[@]}"
do
    echo $i
        python evaluation/generation/eva_generation.py \
            --dataset_name $i \
            --model_name_or_path xxx \
            --max_length 1024 

done