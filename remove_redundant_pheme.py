import json
import re
import os

from tqdm import tqdm



def clean_sentence(text):
    clean_text = re.sub(r'http\S+', '', text)

    return clean_text


if __name__ == '__main__':

    dataset = 'Pheme'

    folder_path = os.path.join('dataset', dataset, 'source')

    files = os.listdir(folder_path)

    for file_name in tqdm(files):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        comments_map = {comment["comment id"]: comment for comment in data["comment"]}

        parent_content = data["source"]["content"]
        parent_content = clean_sentence(parent_content)
        data["source"]["content"] = parent_content

        for comment in data["comment"]:
            comment_content = comment["content"]
            comment_content = clean_sentence(comment_content)
            comment["content"] = comment_content


        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)