array=(
    logs/cherry_alpaca_5_per-VS-alpaca/koala_wrap.json
    logs/cherry_alpaca_5_per-VS-alpaca/lima_wrap.json
    logs/cherry_alpaca_5_per-VS-alpaca/sinstruct_wrap.json
    logs/cherry_alpaca_5_per-VS-alpaca/vicuna_wrap.json
    logs/cherry_alpaca_5_per-VS-alpaca/wizardlm_wrap.json
)
for i in "${array[@]}"
do
    echo $i
        python evaluation/generation/eval.py \
            --wraped_file $i \
            --batch_size 15 \
            --api_key xxx \
            --api_model gpt-4

done