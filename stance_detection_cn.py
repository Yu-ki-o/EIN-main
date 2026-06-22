import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os

token_key = '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'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", token=token_key)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16, token=token_key
    )

    return model, tokenizer

def clean_sentence(text):
    clean_text = re.sub(r'http\S+', '', text)

    return clean_text


def stance_detection(model, tokenizer, source_sentence, response_sentence, parent):
    
    if parent == -1:
        message_template_stage2 = f"""源帖子：'{source_sentence}' \n 回应评论：'{response_sentence}' \n 根据回应评论的内容，鉴定其对源帖子的态度，并选择以下选项之一：回应评论是相信源帖子的：0，回应评论是不相信（或质疑）源帖子的：1。\n 如果回应评论仅仅包含 （'转发微博', '转发微博。', '轉發微博', '轉發微博。'） 或者是只@了某人而没有其他内容，则就认为回应评论是相信源帖子的。\n 你只需要从上述选项中选择一个标签作为最终结果，不需要额外的文字。"""
    else:
        message_template_stage2 = f"""源句子：'{source_sentence}' \n 回应句子：'{response_sentence}' \n 根据回应句子的内容，鉴定其对源句子的态度，并选择以下选项之一：回应句子是同意源句子的：0，回应句子是不同意（或质疑）源句子的：1。\n 如果回应句子仅仅包含 （'转发微博', '转发微博。', '轉發微博', '轉發微博。'） 或者是只@了某人而没有其他内容，则就认为回应是同意源句子的。\n 你只需要从上述选项中选择一个标签作为最终结果，不需要额外的文字。"""


    #######################
    input_text = message_template_stage2
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.2)

    # print(tokenizer.decode(outputs[0]))
    stance_label = int(re.findall(r'\d+', tokenizer.decode(outputs[0]))[-1])

    #######################

    # stage_templates = [message_template_stage1, message_template_stage2]

    # labels = []

    # for messages in stage_templates:

    #     input_text = messages
    #     input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    #     outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.2)

    #     # print(tokenizer.decode(outputs[0]))
    #     reply_label = int(re.findall(r'\d+', tokenizer.decode(outputs[0]))[-1])
    #     labels.append(reply_label)
    
    # if labels[0] == 1:
    #     stance_label = 2
    # else:
    #     stance_label = labels[1]

    return stance_label


if __name__ == '__main__':

    model, tokenizer = load_model()

    dataset = 'Weibo'

    folder_path = os.path.join('dataset', dataset, 'source')

    files = os.listdir(folder_path)

    for file_name in tqdm(files):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        comments_map = {comment["comment id"]: comment for comment in data["comment"]}

        for comment in data["comment"]:
            parent_id = comment["parent"]
            if parent_id == -1:
                source_sentence = data["source"]["content"]
            else:
                source_sentence = comments_map[parent_id]["content"]
            
            reply_sentence = comment["content"]

            # clean sentences before detection
            source_sentence = clean_sentence(source_sentence)
            reply_sentence = clean_sentence(reply_sentence)

            stance_label = stance_detection(model, tokenizer, source_sentence, reply_sentence, parent_id)
            comment["stance_label"] = stance_label

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)