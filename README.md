# PMET
This is a repository for "PMET: Precise Model Editing in a Transformer"

## An example for editing GPT-J (6B) on counterfact dataset using PMET
### 1. Edit GPT-J model
```bash
python evaluate.py --model_path [your model path] --model_name EleutherAI/gpt-j-6B --alg_name PMET --hparams_fname EleutherAI_gpt-j-6B.json --ds_name mcf --num_edits [num]
```
### 2. Summarize the results
```bash
python summarize.py --dir_name=PMET --runs=run_<run1>,run_<run2>
```

## Another example for editing GPT-J (6B) on zsRE dataset using PMET
```bash
python evaluate.py --model_path [your model path] --model_name EleutherAI/gpt-j-6B --hparams_fname EleutherAI_gpt-j-6B-zsre.json --ds_name zsre --num_edits [num]
```
### Then just summarize the results