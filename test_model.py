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

peft_path = "fine_tuning/chatglm-lora.pt"
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True,
    r=32,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
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
    '你在哪里上学？',
    '你喜欢爸爸还是喜欢妈妈？',

    # Copilot的问题
    '你喜欢吃什么？',
    '你喜欢什么运动？',
    '你喜欢什么电影？',
    '你喜欢什么音乐？',
    '你喜欢什么书？',
    '你喜欢什么游戏？',
    '你喜欢什么动漫？',
    '你喜欢什么动物？',
    '你在哪里工作？',
    '你喜欢吃什么？',
    '你喜欢什么颜色？',
    '你喜欢什么水果？',
    '你喜欢什么蔬菜？',
    '你喜欢什么饮料？',
    '你喜欢什么食物？',

    '计算1+1=',
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
    print(out_text)
