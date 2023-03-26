import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer

from modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained(
    "THUDM/chatglm-6b",
    load_in_8bit=True,
    trust_remote_code=True,
    device_map='auto')

peft_path = "chatglm_ypw/adapter_model.bin"
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True,
    r=32,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=[
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
    ],
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = model.eval()

questions = [
    # 我的问题
    '你是谁？',
    '你几岁了？',
    '你上几年级了？',
    '你喜欢爸爸还是喜欢妈妈？',
    '你喜欢吃什么？',
    '你喜欢看什么动画片？',

    # Copilot的问题
    '计算1+1=',
    '请写一个 Python 函数，输入一个整数，返回它的平方，请直接给出代码。',
    '你好，请写一首关于特斯拉的诗。要求押韵，符合七言绝句。'
]

for idx, q in enumerate(questions):
    input_text = f'Human: {q} \n\nAssistant: '
    batch = tokenizer(input_text, return_tensors="pt")
    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=torch.ones_like(batch["input_ids"]).bool(),
        max_length=512,
        top_p=0.7,
        temperature=0.9)
    out_text = tokenizer.decode(out[0])
    answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
    print(out_text.replace('\n\n', '\n'), '\n')
