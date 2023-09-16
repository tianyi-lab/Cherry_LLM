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
        python evaluation/generation/eval_generation_wrap.py \
            --dataset_name $i \
            --fname1 cherry_alpaca_5_per/test_inference \
            --fname2 alpaca/test_inference \
            --save_name cherry_alpaca_5_per-VS-alpaca \
            --max_length 1024
done

