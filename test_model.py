import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer

from modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained(
    "THUDM/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map='auto')

peft_path = "fine_tuning/chatglm-lora.pt"
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = model.eval()

questions = [
    '计算1+1=',
    '你是谁？',
    '你好，请写一首关于特斯拉的诗。要求押韵，符合七言绝句。'
]

for q in questions:
    result = model.chat(tokenizer, query=q, history=[])
    print(result)

for idx, q in enumerate(questions):
    input_text = f'Human: {q} \n\nAssistant: '
    batch = tokenizer(input_text, return_tensors="pt")
    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=torch.ones_like(batch["input_ids"]).bool(),
        max_length=512,
        top_p=0.7,
        temperature=0.9
    )
    out_text = tokenizer.decode(out[0])
    answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
    print(q, out_text)
