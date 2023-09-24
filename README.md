
# From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning


[From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/abs/2308.12032)

<p align="center" width="40%">
<a ><img src="images/cherry.jpeg" alt="overview" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>


This is the repo for the Cherry Data Selection project, which introduces a self-guided methodology for LLMs to autonomously discern and select cherry samples from vast open-source datasets, effectively minimizing manual curation and potential cost for instruction tuning an LLM.

The repo contains:

- The cherry data used for fine-tuning the model, cherry_data_v1 represents the cherry data obtained based on the llama-1 model. 
- The model checkpoints (7B) that were trained using our cherry data.
- The code for selecting cherry data from the existing instruction-tuning dataset.


## News
- [2023/09] We partially reconstructed the repo structure and added some results on llama2.  
- [2023/09] We released codes for evaluating the performance between two LLMs by using GPT4 or chatGPT. 
- [2023/09] We released codes for this project.

## Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Install](#install)
- [Run Code](#run-code)
- [Data and Model Weights V1](#data-and-model-weights-v1)
- [Evaluation](#evaluation)
- [Performance Comparison ](#performance-comparison)
- [Prompt](#prompt)
- [Hyperparameters](#hyperparameters)
- [ToDo](#todo)
- [Citation](#citation)

## Overview

Our study puts forth a method for autonomously sifting through expansive open-source datasets to discover the most impactful training samples. We coin these samples as "cherry data", designating those data fragments that hold the potential to exponentially enhance LLM instruction tuning. At the heart of our research is the hypothesis that during their preliminary training stages with carefully chosen instruction data, LLMs can develop an intrinsic capability to discern instructions. This foundational understanding equips them with the discernment to assess the quality of broader datasets thus making it possible to estimate the instruction-following difficulty in a self-guided manner. 

<p align="center" width="70%">
<a ><img src="images/method_overview.png" alt="overview" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

Initially, the model is familiarized with a fraction of the target dataset during the "Learning from Brief Experience" phase. This preliminary knowledge paves the way for the subsequent "Evaluating Based on Experience" phase, where we meticulously evaluate the model's response generation. To estimate the difficulty of a given example, we propose a novel metric called Instruction-Following Difficulty (IFD) score in which both models' capability to generate a response to a given instruction and the models' capability to generate a response directly are measured and compared. By calculating Instruction-Following Difficulty (IFD) scores, we quantify the challenge each sample presents to the model. Harnessing these insights, the "Retraining from Self-Guided Experience" phase utilizes cherry data with standout IFD scores to hone the model, culminating in our superior cherry models. The net result is a model that aligns more adeptly with instructions, ensuring enhanced performance.

## Highlights

* The selection of cherry data in this project is entirely self-guided and does not need ANY extra outside models, ranging from BERT to chatGPT.
* We use approximately 5% or 10% of the data to have comparable performances to the models trained on full data, which is experimented on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [WizardLM](https://github.com/nlpxucan/WizardLM) datasets.
* The IFD score provided by us can divide the samples into better or relatively bad ones, which might provide insight into the types of data good for instruction tuning.

## Install

Install the dependencies with `pip install -r requirements.txt`

Note: This `requirements.txt` is originated from the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). If you are using a different code base with PyTorch installed, we recommend you manually install the below packages and do not need to install from `requirements.txt`

`pip install tqdm`

`pip install scikit-learn`

## Run Code


1. Select Pre-Experienced Data

```
python cherry_seletion/data_analysis.py \
    --data_path data/alpaca_data.json \
    --save_path alpaca_data_pre.pt \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --max_length 512 \
    --prompt alpaca \
    --mod pre
```

```--data_path```: The targeted dataset in the Alpaca format <br>
```--save_path```: The path to save the ```.pt``` file containing embeddings or scores <br>
```--prompt```: The prompt type used for training and selecting data, can choose between ```alpaca``` or ```wiz``` <br>
```--mod```: ```pre``` used for getting needed embeddings or scores on selecting pre-experienced samples and ```cherry``` used for cherry <br>

```
python cherry_seletion/data_by_cluster.py \
    --pt_data_path alpaca_data_pre.pt \
    --json_data_path data/alpaca_data.json \
    --json_save_path alpaca_data_pre.json \
    --sample_num 10 \
    --kmeans_num_clusters 100 \
    --low_th 25 \
    --up_th 75
```

```--pt_data_path```: The ```.pt``` file from previous step containing needed embeddings or scores
```--json_data_path```: The targeted dataset in the Alpaca format <br>
```--json_save_path```: The path to save the selected pre-experienced samples <br>
```--sample_num```: How many samples will be selected in each cluster <br>
```--kmeans_num_clusters```: How many clusters will be generated by K-Means <br>
```--low_th``` and ```--up_th```: The lower and Upper threshold for selecting samples within each cluster <br>


3. Train Pre-Experienced Model

4. Select Cherry Data

```
python cherry_seletion/data_analysis.py \
    --data_path data/alpaca_data.json \
    --save_path alpaca_data_cherry.pt \
    --model_name_or_path <your_path_pre_experienced_model> \
    --max_length 512 \
    --prompt alpaca \
    --mod cherry
```

```
python cherry_seletion/data_by_IFD.py \
    --pt_data_path alpaca_data_cherry.pt \
    --json_data_path data/alpaca_data.json \
    --json_save_path alpaca_data_cherry.json \
    --max_length 512 \
    --sample_rate 0.06 \
    --prompt alpaca
```

```--sample_rate```: How many cherry samples you would like to select? You can also use ```--sample_number``` to set the exact number of samples. 

6. Train Cherry Model

## Data and Model Weights V1

The following table provides a comparison between our cherry models and baseline models on the Huggingface Open LLM Leaderboard and AlpacaEval Leaderboard. 
These results are based on cherry_data_v1. The prompt and training hyperparameters can be found in the Hyperparameters section. 
These results verify the effectiveness of our method, which can be used to select the most valuable data samples for instruction tuning. 


|                          | **Avg** | **ARC** | **HellaSwag** | **MMLU** | **TruthfulQA** || **AlpacaEval** ||**Data**| **Model**|
|--------------------------|:-----------:|:-------:|:-------------:|:-------:|:--------------:|:-:|:--------------:|:-:|:-:|:-:|
| **Alpaca**      | 50.21       | 42.65   | 76.91         | 41.73   | 39.55          || 26.46          ||/|/|
| **5% Alpaca**     | 52.06| 53.92   | 79.49         | 36.51   | 38.33          || 34.74          ||[[Link]](data/cherrt_alpaca/cherry_alpaca_5_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-alpaca-5-percent-7B)|
| **10% Alpaca**     | /       | /   | /         | /   | /          || /          ||[[Link]](data/cherrt_alpaca/cherry_alpaca_10_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-alpaca-10-percent-7B)|
| **15% Alpaca**     | /       | /   | /         | /   | /          || /          ||[[Link]](data/cherrt_alpaca/cherry_alpaca_15_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-alpaca-15-percent-7B)|
||||||||||||
| **WizardLM**    | 54.18       | 51.60   | 77.70         | 42.70   | 44.70          || 67.64          ||/|/|
| **WizardLM*** | 52.79  | 53.07   | 77.44         | 37.75   | 42.90          || 61.99          ||[[hf-Link]](https://huggingface.co/datasets/MingLiiii/cherry_wizardlm_filtered)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-wizardlm-filtered-7B)|
| **10% WizardLM**  | 51.59       | 52.90   | 78.95         | 33.08   | 41.41         || 61.44          ||[[Link]](data/cherry_wizardLM/cherry_wizardLM_10_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-wizardlm-10-percent-7B)|
| **20% WizardLM**     | /       | /   | /         | /   | /          || /          ||[[Link]](data/cherry_wizardLM/cherry_wizardLM_20_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-wizardlm-20-percent-7B)|
| **20% WizardLM**     | /       | /   | /         | /   | /          || /          ||[[Link]](data/cherry_wizardLM/cherry_wizardLM_30_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-wizardlm-30-percent-7B)|
| **40% WizardLM**  | 52.83       | 53.07   | 77.79         | 35.29   | 45.17          || 65.09          ||[[Link]](data/cherry_wizardLM/cherry_wizardLM_40_percent.json)|[[hf-Link]](https://huggingface.co/MingLiiii/cherry-wizardlm-40-percent-7B)|
||||||||||


Also, the WizardLM filter script is provided here: [[Link]](cherry_seletion/filter.py)

Note: Due to the hardware limit, all our models are using the 7B model. 

## Evaluation

We release the codes and data for using GPT4 or chatGPT to evaluate and compare the performance between two LLMs. This method greatly eliminates the potential position bias of GPT4 and chatGPT. For details, please see [AlpaGasus](https://github.com/Lichang-Chen/AlpaGasus) or our [paper](https://arxiv.org/abs/2308.12032). We thank [@Lichang-Chen](https://github.com/Lichang-Chen) and [AlpaGasus](https://github.com/Lichang-Chen/AlpaGasus) repo for sharing the evaluation codes.  

To use this code, please follow the below scripts:

```bash scripts/do_eval_generation.sh```: The model automatically generates the responses for a given instruction in test datasets. <br>
```bash scripts/do_eval_generation_wrap.sh```: Wrap the response files of LLMs being compared. <br>
```bash scripts/do_eval.sh```: Use GPT4 or chatGPT for the evaluation. <br>
```bash scripts/do_review_eval_score.sh```: Parse the results and draw the figure. <be>

More detailed illustrations will be updated. Feel free to drop me an email if you are urgent about it. 

## Performance Comparison 

Comparing our models trained on selected data with models trained on full data. (a) Comparison between our model with 5% Alpaca data and the official Alpaca model. (b) Comparison between our model with 10% WizardLM data and the reimplemented WizardLM model. (c) Comparison between our model with 40% WizardLM data and the official WizardLM model. All these experiments use GPT4 as the judge. Each horizontal bar represents a comparison in a specific test set. 

<p align="center" width="100%">
<a ><img src="images/main_result_gpt4.png" alt="overview" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>


## Prompt

We used the following prompts for fine-tuning the cherry models with Alpaca data:

- for examples with a non-empty input field:

 ```
 Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
 
 ### Instruction:
 {instruction}
 
 ### Input:
 {input}
 
 ### Response:
 ```

- for examples with an empty input field:

 ```
 Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
 ### Instruction:
 {instruction}
 
 ### Response:
 ```

We used the following prompts for fine-tuning the cherry models with Wizard data:

```
{instruction}

### Response:
```

## Hyperparameters

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay | Warmup Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cherry Models (Alpaca) | 128 | 2e-5 | 3 | 512 | 0 | 0.03 |
| Cherry Models (WizardLM) | 128 | 2e-5 | 3 | 1024 | 0 | 0.03 |

## ToDo
- [x] Release the code, data, and models. 
- [x] Release the evaluation code for comparison.
- [ ] Train Cherry WizardLM with the length of 2048.
- [ ] Maybe try using QLORA.

## Citation

Please consider citing our paper if you think our codes, data, or models are useful. Thank you!
```
@misc{li2023quantity,
      title={From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning}, 
      author={Ming Li and Yong Zhang and Zhitao Li and Jiuhai Chen and Lichang Chen and Ning Cheng and Jianzong Wang and Tianyi Zhou and Jing Xiao},
      year={2023},
      eprint={2308.12032},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```










