import torch
import json
import numpy as np
import argparse
from tqdm import tqdm

PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_WIZARDLM = {
    "prompt_input": (
        "{instruction}\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_VICUNA = {
    "prompt_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction}\nInput:\n{input} ASSISTANT:"
    ),
    "prompt_no_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=float, default=0)
    parser.add_argument("--sample_number", type=int, default=0)
    parser.add_argument("--prompt", type=str, default='alpaca', help='vicuna, wiz, alpaca')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    prompt_specific_len = 0
    if args.prompt == 'alpaca':
        prompt_no_input = PROMPT_DICT_ALPACA["prompt_no_input"]
        prompt_input = PROMPT_DICT_ALPACA["prompt_input"]
        direct_answer_prompt = '### Response:'
        prompt_specific_len = 4
    elif args.prompt == 'wiz':
        prompt_no_input = PROMPT_DICT_WIZARDLM["prompt_no_input"]
        prompt_input = PROMPT_DICT_WIZARDLM["prompt_input"]
        direct_answer_prompt = '### Response:'
        prompt_specific_len = 4
    elif args.prompt == 'vicuna':
        prompt_no_input = PROMPT_DICT_VICUNA["prompt_no_input"]
        prompt_input = PROMPT_DICT_VICUNA["prompt_input"]
        direct_answer_prompt = 'ASSISTANT:'
        prompt_specific_len = 6

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    new_data = []
    for i in tqdm(range(len(pt_data))):

        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        json_data_i = json_data[i]
        instruct_i = json_data_i['instruction']
        output_i = json_data_i['output']

        direct_answer_text = direct_answer_prompt + output_i
        input_i = json_data_i['input'] if 'input' in json_data_i.keys() else ''
        if input_i == '':
            temp_dict = {'instruction':instruct_i}
            promt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use
        else:
            temp_dict = {'instruction':instruct_i,'input':input_i}
            promt_to_use = prompt_input.format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use

        # Tokenize the input text
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to('cpu')
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):

            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = tokenizer.encode(text_temp)
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]

            loss_list = loss_list_[start_token-1:end_token_real-1] 

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if args.max_length-instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i, args.max_length-instruct_i_len+prompt_specific_len, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, args.max_length, loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                mean_rate = np.nan
                json_data_i['ifd_score'] = mean_rate
                new_data.append(json_data_i)
                continue

            if instruct_i_len + len_1 > args.max_length:
                mean_rate = np.nan
                json_data_i['ifd_score'] = mean_rate
                new_data.append(json_data_i)
                continue

            mean_1 = loss_list_1.mean()
            mean_2 = loss_list_2.mean()
            mean_rate = mean_2/mean_1
            # if mean_rate > 1: 
            #     continue

            mean_rate_list.append((mean_rate,i))
            mean_list_1.append((mean_1,i))
            mean_list_2.append((mean_2,i))

            json_data_i['ifd_score'] = mean_rate
            new_data.append(json_data_i)

        else:
            mean_rate = np.nan
            json_data_i['ifd_score'] = mean_rate
            new_data.append(json_data_i)
            continue

    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)


if __name__ == '__main__':
    main()