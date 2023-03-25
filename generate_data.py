import jsonlines
import random

with jsonlines.open('data/belle_open_source_1M.train.json') as data:
    data = list(data)

print(len(data))

random.seed(42)
random.shuffle(data)

print(data[:5])

print(data[0]['input'])

my_data = [
    {'context': '你是谁？\n', 'target': '我是杨开心。'},
    {'context': '你几岁了？\n', 'target': '我两岁半了。'},
    {'context': '你喜欢吃什么？\n', 'target': '我喜欢吃雪糕。'},
    {'context': '你喜欢看什么动画片？\n', 'target': '我喜欢看小猪佩奇。'},
    {'context': '你喜欢爸爸还是喜欢妈妈？\n', 'target': '我都喜欢。'},
]

copy_times = 500
data3 = data[:10000 - len(my_data) * copy_times] + my_data * copy_times

data4 = [
    x if 'context' in x else
    {'context': x['input'], 'target': x['target']}
    for x in data3
]

print(len(data4), len(my_data) * copy_times)

random.seed(42)
random.shuffle(data4)

with jsonlines.open('data/ypw_data.jsonl', 'w') as f:
    f.write_all(data4)
