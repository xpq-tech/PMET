# PMET
- Code for [``PMET: Precise Model Editing in a Transformer``](https://arxiv.org/abs/2308.08742)

## Requirements
- At least one A100/A800 80G GPU and another GPU with no less than 24G memory.
- Environment
    ``` bash
    conda create -n pmet python=3.10
    pip install -r requirements.txt
    ```

## Quick Start
### An example for editing GPT-J (6B) on counterfact dataset using PMET
#### 1. Edit GPT-J model 
 
    python evaluate.py --model_path [your model path] --model_name EleutherAI/gpt-j-6B --alg_name PMET --hparams_fname EleutherAI_gpt-j-6B.json --ds_name mcf --num_edits [num]


 Computing $C_0$ and $v^m_i$ for 10K edits will take about a day. But the $C_0$ and $v^m_i$ for the 10K edits will be stored after computing, they will be directly used without recomputing in the next run for the 10K edits. It will then take about 1 hour to complete the model editing. After model editing, it will take a dozen hours to complete the evaluation. Using '--skip_generation_tests' will significantly speed up evaluation.
#### 2. Summarize the results

    python summarize.py --dir_name=PMET --runs=run_<run1>,run_<run2>


### Another example for editing GPT-J (6B) on zsRE dataset using PMET

    python evaluate.py --model_path [your model path] --model_name EleutherAI/gpt-j-6B --hparams_fname EleutherAI_gpt-j-6B-zsre.json --ds_name zsre --num_edits [num]

#### Then just summarize the results

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git).

## Citation

Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, Jie Yu.
PMET: Precise Model Editing in a Transformer.
arXiv preprint arXiv:2308.08742 (2023).

```
@article{li2023pmet,
  title={PMET: Precise Model Editing in a Transformer},
  author={Li, Xiaopeng and Li, Shasha and Song, Shezheng and Yang, Jing and Ma, Jun and Yu, Jie},
  journal={arXiv preprint arXiv:2308.08742},
  year={2023}
}
```