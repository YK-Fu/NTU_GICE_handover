import time
import json
import requests

port_num = 8889
headers = {"Content-Type": "application/json"}

def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data),
                        headers=headers)

    sentences = resp.json()['sentences']
    return sentences

## test
max_attempts = 50
attempt = 0
wait_time = 20  # 等待時間，以秒為單位
reponsed = False

data = {"sentences": [''],
        "tokens_to_generate": 2,
        "temperature": 0.9,
        "add_BOS": True,
        "top_k": 0,
        "top_p": 0.9,
        "greedy": False,
        "all_probs": False,
        "repetition_penalty": 1.2,
        "min_tokens_to_generate": 2,
        }

dots = [".  ", ".. ", "..."]
emojis = ['🫠','😉','😇','🥰','🤩','😘','☺️','😗','🤪','😋','🤪','🤗','🤫','🤮','🙄','😏','😵','😳','🥺','😭']
# ['🤗','😉','🥰','😵','🤫','🤪','☺️']

while attempt < max_attempts and not reponsed:
    try:
        sentences = request_data(data)
        reponsed = True
    except requests.exceptions.ConnectionError:
        # 在重試之前等待
        for i in range(wait_time*5):
            idx_1 = i % 3
            idx_2 = i % 20
            time.sleep(0.2)  # 等待指定的間隔時間
            print(f"Connection failed. Model is still loading{dots[idx_1]} Please wait {emojis[idx_2]}", end="\r", flush=True)
        attempt += 1

print("**********************************")
print("您現在可以開始與聊天機器人對話了!")
print("**********************************")

while True:
    question = input('User: ')
    # prompt = f"<|start_header_id|>system<|end_header_id|>\n\nA and B are chat casually. Modality: \{User: text, Machine: text\}.<|eot_id|><|start_header_id|>A<|end_header_id|>{question}<|eot_id|><|start_header_id|>B<|end_header_id|>\n\n"
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are a chatbot, only response precisely. Modality: {{User: text, Machine: text}}.<|eot_id|>\n<|start_header_id|>User<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>Machine<|end_header_id|>\n\n"

    data = {"sentences": [prompt],
        "tokens_to_generate": 200,
        "temperature": 1,
        "add_BOS": True,
        "top_k": 20,
        "top_p": 1,
        "greedy": False,
        "all_probs": False,
        "repetition_penalty": 1,
        "min_tokens_to_generate": 1,
        "end_strings": ['<|eot_id|>', '<|end_of_text|>']
        }
    # breakpoint()
    response = request_data(data)[0]
    print('Chatbot:', response)

