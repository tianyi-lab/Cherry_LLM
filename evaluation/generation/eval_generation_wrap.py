import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='', help="The name of the dataset to use.")
    parser.add_argument("--fname1", type=str, default='')
    parser.add_argument("--fname2", type=str, default='')
    parser.add_argument("--save_name", type=str, default='') # a vs b format
    parser.add_argument("--max_length", type=int, default=1024)

    args = parser.parse_args()
    return args

args = parse_args()

print('args.dataset_name',args.dataset_name)
f_name = args.dataset_name+'_'+str(args.max_length)+'.json'
args.fname1 = os.path.join(args.fname1,f_name)
args.fname2 = os.path.join(args.fname2,f_name)
print('args.fname1',args.fname1)
print('args.fname2',args.fname2)
with open(args.fname1, "r") as f:
    data1 = json.load(f)
with open(args.fname2, "r") as f:
    data2 = json.load(f)

saved_info = {'Meta_Info':{'dataset_name':args.dataset_name,'fname1':args.fname1,'fname2':args.fname2}}

assert len(data1) == len(data2)

if(args.dataset_name=="vicuna"):
    prompt_key = 'text'
elif(args.dataset_name=="koala"):
    prompt_key = 'prompt'
elif(args.dataset_name=="sinstruct"):
    prompt_key = 'instruction'
elif(args.dataset_name=="wizardlm"):
    prompt_key = 'Instruction'
elif(args.dataset_name=="lima"):
    prompt_key = 'conversations'

new_data = []
for i, sample_i in enumerate(data1):
    new_sample = {}
    new_sample[prompt_key] = sample_i[prompt_key]
    new_sample['Answer1'] = data1[i]['response']
    new_sample['Answer2'] = data2[i]['response']
    new_data.append(new_sample)

print('New data len \n',len(new_data))
saved_info['data'] = new_data

save_path = os.path.join('logs',args.save_name,args.dataset_name+'_wrap.json')
os.makedirs(os.path.join('logs',args.save_name), exist_ok=True)
with open(save_path, "w") as fw:
    json.dump(saved_info, fw, indent=4)

pass
