# ChatGLM-Tuning

一种平价的chatgpt实现方案，基于清华的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) + LoRA 进行finetune.

数据集: [alpaca](https://github.com/tatsu-lab/stanford_alpaca)

有colab的同学可以直接在colab上尝试： <a href="https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/finetune.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>

## YPW

初步训练成果，让ChatGLM保持原有能力的同时回答我是谁，我多大了，我喜欢吃什么……

<img width="979" alt="image" src="https://user-images.githubusercontent.com/10473170/227542465-73a5074c-b84a-4ba3-914f-350e8c91cdd4.png">

<img width="787" alt="image" src="https://user-images.githubusercontent.com/10473170/227542504-c8ec33ae-d2d6-4bdd-951b-56759e5ff5e7.png">

<img width="556" alt="image" src="https://user-images.githubusercontent.com/10473170/227542523-03ebaee1-c39d-424c-abbd-e1cba6c94b66.png">

## S1 Finetune

### 准备

- 显卡: 显存 >= 16G (最好24G或者以上)
- 环境：
- - python>=3.8
- - cuda>=11.6, cupti, cuDNN, TensorRT等深度学习环境
- - pip3 install -r requirements.txt


### 数据预处理


转化alpaca数据集为jsonl

```bash
python cover_alpaca2jsonl.py \
    --data_path data/alpaca_data.json \
    --save_path data/alpaca_data.jsonl \
```

tokenization

```bash
python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 200 \ 
    --skip_overlength
```

- `--jsonl_path` 微调的数据路径, 格式jsonl, 对每行的['context']和['target']字段进行encode
- `--save_path` 输出路径
- `--max_seq_length` 样本的最大长度

### 训练

```bash
python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
```

### 推理

参考 [infer.ipynb](infer.ipynb)

<details><summary><b>Finetune前后对比</b></summary>

利用Alpaca数据集合对ChatGLM-6B Finetune后，在Alpaca数据集上表现得更好:
- `Answer:` 是模型的输出
- `#### Answer:` 是原答案
![](https://user-images.githubusercontent.com/6883957/226977555-c00c796f-4fdb-4613-810a-8b9a6068bb1b.jpeg)


</details>


## S2. Reward Model

## S3. PPO


## LoRA

| LoRA      | Dataset |
| ----------- | ----------- |
| mymusise/chatglm-6b-alpaca-lora      | Alpaca       |
| *(on the way)*      | Alpaca-zh       |

### 使用预训练好的LoRA

参考 [examples/infer_pretrain.ipynb](https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/infer_pretrain.ipynb)


# TODO:

- ~~bs > 1 support~~
- 使用中文数据
- 加入RLHF
