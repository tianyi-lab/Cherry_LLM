import json
import argparse

def filter_and_sort(data, num_select):
    filtered_data = [item for item in data if item['ifd_score'] <= 1]
    
    sorted_data = sorted(filtered_data, key=lambda x: x['ifd_score'], reverse=True)
    
    return sorted_data[:num_select]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--sample_rate", type=float, default=0)
    parser.add_argument("--sample_number", type=int, default=0)
    args = parser.parse_args()

    with open(args.json_data_path, 'r') as f:
        data = json.load(f)

    if args.sample_number == 0:
        args.sample_number = int(len(data)*args.sample_rate)

    new_data = filter_and_sort(data, args.sample_number)

    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)

if __name__ == "__main__":
    main()
