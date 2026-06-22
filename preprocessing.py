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

    
        comments_map = {comment["comment id"]: comment for comment in data["comment"]}

        onhold_comments = []
        for comment in data["comment"]:
            parent_id = comment["parent"]
            if parent_id == -1:
                if comment["stance_label"] == 0:
                    comment["state"] = 0
                elif comment["stance_label"] == 1:
                    comment["state"] = 1
            
                comment["hop"] = 1 # add hop label

            else:
                value = comments_map[parent_id].get('state', None)
                if value is None:
                    onhold_comments.append(comment)
                    continue
                else:
                    parent_state = comments_map[parent_id]["state"]

                    if parent_state == 0 and comment["stance_label"] == 0:
                        comment["state"] = 0
                    elif parent_state == 0 and comment["stance_label"] == 1:
                        comment["state"] = 1
                    elif parent_state == 1 and comment["stance_label"] == 0:
                        comment["state"] = 1
                    elif parent_state == 1 and comment["stance_label"] == 1:
                        comment["state"] = 0
                
                    comment["hop"] = comments_map[parent_id]["hop"] + 1

    
        onhold_comments2 = []
        for comment in onhold_comments:
            parent_id = comment["parent"]
            
            value = comments_map[parent_id].get('state', None)
            if value is None:
                onhold_comments.append(comment)
            else:
                parent_state = comments_map[parent_id]["state"]

                if parent_state == 0 and comment["stance_label"] == 0:
                    comment["state"] = 0
                elif parent_state == 0 and comment["stance_label"] == 1:
                    comment["state"] = 1
                elif parent_state == 1 and comment["stance_label"] == 0:
                    comment["state"] = 1
                elif parent_state == 1 and comment["stance_label"] == 1:
                    comment["state"] = 0
                
                comment["hop"] = comments_map[parent_id]["hop"] + 1
        
        for comment in onhold_comments2:
        
            parent_id = comment["parent"]
            parent_state = comments_map[parent_id]["state"]

            if parent_state == 0 and comment["stance_label"] == 0:
                comment["state"] = 0
            elif parent_state == 0 and comment["stance_label"] == 1:
                comment["state"] = 1
            elif parent_state == 1 and comment["stance_label"] == 0:
                comment["state"] = 1
            elif parent_state == 1 and comment["stance_label"] == 1:
                comment["state"] = 0
            
            comment["hop"] = comments_map[parent_id]["hop"] + 1

        # add state attribute to json
        hop_counts = {}
        for comment in data['comment']:
            hop = comment['hop']
            state = comment['state']
            if hop not in hop_counts:
                hop_counts[hop] = {'state_0': 0, 'state_1': 0}
            if state == 0:
                hop_counts[hop]['state_0'] += 1
            elif state == 1:
                hop_counts[hop]['state_1'] += 1

        # Adding the 'state' attribute to the main dictionary
        data['state'] = {f"{hop}-hop": hop_counts[hop] for hop in sorted(hop_counts)}
  
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)