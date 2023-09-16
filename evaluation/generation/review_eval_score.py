import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_home_path", type=str, default='', help="home path that save the reviews")
    parser.add_argument('--task_list', nargs='+', type=str, default=['Vicuna','Koala','WizardLM','SInstruct','LIMA'])
    parser.add_argument("--key1", type=str, default='Model1')
    parser.add_argument("--key2", type=str, default='Model2')
    parser.add_argument("--save_name", type=str, default='result') # a vs b format
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo')

    args = parser.parse_args()
    return args

args = parse_args()


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlBu'](
        np.linspace(0.2, 0.8, data.shape[1]))

    fig, ax = plt.subplots(figsize=(8, 5),dpi=200)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.6,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'black'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=1, loc='upper right', fontsize='medium')

    return fig, ax


results = {}
def get_scores_all(pure_data):
    score1, score2, score3 = 0, 0, 0
    l = len(pure_data)
    for i in range(l):
        k1_score = eval(pure_data[i]['scores'])[0]
        k2_score = eval(pure_data[i]['scores'])[1]
        k1_score_reverse = eval(pure_data[i]['scores_reverse'])[1]
        k2_score_reverse = eval(pure_data[i]['scores_reverse'])[0]

        if k1_score > k2_score and k1_score_reverse > k2_score_reverse:
            score1 += 1
        elif k1_score < k2_score and k1_score_reverse > k2_score_reverse:
            score2 += 1
        elif k1_score > k2_score and k1_score_reverse < k2_score_reverse:
            score2 += 1
        elif k1_score == k2_score and k1_score_reverse > k2_score_reverse:
            score1 += 1
        elif k1_score > k2_score and k1_score_reverse == k2_score_reverse:
            score1 += 1
        elif k1_score == k2_score and k1_score_reverse < k2_score_reverse:
            score3 += 1
        elif k1_score < k2_score and k1_score_reverse == k2_score_reverse:
            score3 += 1
        elif k1_score == k2_score and k1_score_reverse == k2_score_reverse:
            score2 += 1
        elif k1_score < k2_score and k1_score_reverse < k2_score_reverse:
            score3 += 1
    return [score1, score2, score3]

for dataset in args.task_list:
    review_path = ''
    for root, ds, fs in os.walk(args.review_home_path):
            for f in fs:
                if 'gpt-3.5' in args.api_model:
                    if 'reviews_gpt3.5' in f and f.endswith('.json') and dataset.lower() in f:
                        review_path = os.path.join(root, f)
                elif 'gpt-4' in args.api_model:
                    if 'reviews_gpt4' in f and f.endswith('.json') and dataset.lower() in f:
                        review_path = os.path.join(root, f)
    with open(review_path, "r") as f:
        review_data = json.load(f)
    pure_data = review_data['data']

    scores = get_scores_all(pure_data)
    category_names = [f"{args.key1} wins", "Tie", f"{args.key2} wins"]
    results[dataset] = scores

def cal_rate(results):
    win = 0
    tie = 0
    loss = 0
    for k in results.keys():
        win += results[k][0]
        tie += results[k][1]
        loss += results[k][2]
    print((win-loss)/(win+loss+tie)+1)

cal_rate(results)
survey(results, category_names)
img_path = os.path.join(args.review_home_path,args.save_name+'.jpg')
plt.savefig(img_path)
pass
