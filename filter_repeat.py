import json
import os
from tqdm import tqdm

if __name__ == '__main__':

    dataset = 'DRWeibo'

    folder_path = os.path.join('dataset', dataset, 'source')

    files = os.listdir(folder_path)

    for file_name in tqdm(files):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        filtered_comments = [comment for comment in data['comment'] if not (comment['parent'] in comment['children'])]

        # 更新JSON数据
        data['comment'] = filtered_comments

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)