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
    
    # postive:0 negative:1
    if parent == -1:
        message_template_stage2 = f"""Source post: '{source_sentence}' \n Response comment: '{response_sentence}' \n Based on the content of the response comment, determine its attitude towards the source post and choose one of the following options: The response comment believes the source post:0, The response comment do not believes (or doubts) the source post:1. \n If the response comment only contains '@' someone(s) without any other content, then you can consider that the response is believing the source post. \n You only need to select one label from the options above as the final result, no additional text is required."""
    else:
        message_template_stage2 = f"""Source sentence: '{source_sentence}' \n Response sentence: '{response_sentence}' \n Based on the content of the response sentence, determine its attitude towards the source sentence and choose one of the following options: The response sentence agrees the source sentence:0, The response sentence disagrees (or doubts) the source sentence:1. \n If the response sentence only contains '@' someone(s) without any other content, then you can consider that the response is agreeing the source sentence. \n You only need to select one label from the options above as the final result, no additional text is required. """


    #######################
    input_text = message_template_stage2
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.2)

    stance_label = int(re.findall(r'\d+', tokenizer.decode(outputs[0]))[-1])

    #######################

    return stance_label


if __name__ == '__main__':

    model, tokenizer = load_model()

    dataset = 'Pheme'

    folder_path = os.path.join('dataset', dataset, 'source')

    files = os.listdir(folder_path)

    for file_name in tqdm(files):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as file:
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

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)