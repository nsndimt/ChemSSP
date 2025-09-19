# Rapid Adaptation of Chemical Named Entity Recognition using Few-Shot Learning and LLM Distillation.

### Installation required Python package
Install [miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and then install required package in a new conda environment
```
conda create --name fewshotner python=3.10 -y
conda activate fewshotner
conda install -y numpy scipy pandas scikit-learn jupyter seaborn matplotlib
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -y transformers tokenizers wandb deepspeed pytorch-lightning seqeval tiktoken openai
```

### Download all required code and data
1. Download all python codes and compressed files from [zenodo](https://zenodo.org/record/sadasd)
2. Decompress each file with `tar xf [folder].tar.xz`
3. Check if the root folder contains all the following folders and codes
```
fewshotNER/
├── chatGPT/
├── claude/
├── data/
├── episode/
├── gemini/
├── models/
├── bash_script.py
├── CDEv2_tokenize.py
├── Mat2Vec.py
├── fs_chatgpt_utils.py
├── fs_gemini_utils.py
├── fs_claude_utils.py
├── fs_ner_train.py
├── fs_ner_model.py
├── fs_ner_utils.py
``` 

### Expriments with Human Annotation
- Prepare Data
    - (Optional) Sampling episodes using `fs_ner_utils.py`
    - Using exsiting ones in `episode`
- Train and Test
    - Generate bash command using `bash_script_1.py`
    - Run generated bash file to batch excute expriments with different seed and dataset split

### Expriments with LLM Annotation
- Prepare Data
    - (Optional) Generate LLM annotation and sampling episodes using `fs_chatgpt_utils.py`, `fs_claude_utils.py`, `fs_gemini_utils.py`; 
        - name your sampled paragraphs as `sampled_filterd_paragraph_chunk.jsonl` and put it under project root path
        - please provide your own API key by change codes
        - use following command line arguments in sequence:
            - `--create_task` to create task
            - `--call_api` to call API
            - `--parse_result` to parse LLM response
            - `--sample_episode` to sample
    - Copy `UniNER_N10_K10_Q10_top_500_alpha_0.5.jsonl` from one of the LLM folders(`chatGPT/`, `claude/`, `gemini/`) to `episode/`
- Train and Test
    - Generate bash command using `bash_script_2.py`
    - Run generated bash file to batch excute expriments with different seed and LLM annotation

### Contact
Please create an issue or email to [zhangyue@udel.edu](mailto:zhangyue@udel.edu) should you have any questions.
