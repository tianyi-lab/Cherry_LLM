import os
import json
import numpy as np
import matplotlib.pyplot as plt

review_home_path = 'logs/xxx1-VSxxx2'
datasets = ['Vicuna','Koala','WizardLM','SInstruct','LIMA']
# datasets = ['Vicuna','Koala','WizardLM','SInstruct']

save_name = review_home_path.split('/')[-1]

key1, key2 = save_name.split('-VS-')[0],save_name.split('-VS-')[1]
title_ = save_name

# key1 = 'Pre-Experienced Selected by Alpaca (15%)'
# # key2 = 'WizardLM' + r"$^*$" + '(100%)'
# key2 = 'Alpaca (100%)'
# title_ = key1 + ' vs. ' + key2 


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

for dataset in datasets:
    review_path = ''
    for root, ds, fs in os.walk(review_home_path):
            for f in fs:
                if 'reviews' in f and f.endswith('.json') and dataset.lower() in f:
                    review_path = os.path.join(root, f)
                # if 'reviews_gpt4' in f and f.endswith('.json') and dataset.lower() in f:
                #     review_path = os.path.join(root, f)
    with open(review_path, "r") as f:
        review_data = json.load(f)
    pure_data = review_data['data']

    scores = get_scores_all(pure_data)
    category_names = [f"{key1} wins", "Tie", f"{key2} wins"]
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
img_path = os.path.join(review_home_path,save_name+'.jpg')
plt.title(title_)
plt.savefig(img_path)
pass

# from PIL import Image
# def crop_edges(image_path, left, upper, right, lower):
#     with Image.open(image_path) as img:
#         width, height = img.size
#         cropped = img.crop((left, upper, width - right, height - lower))
#         return cropped
# cropped_img = crop_edges(img_path,45,45,45,45)
# cropped_img.save(img_path)
# pass

