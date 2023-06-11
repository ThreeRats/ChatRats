import json
from tqdm import tqdm
from typing import Any, List



def json_to_txt(json_file):
    
    json_list = json.load(open(json_file, 'r', encoding='utf-8'))
    result_list = []

    for json_dict in tqdm(json_list):
        question = json_dict['question']
        answer = json_dict['answer']
        combine_text = ''
        if question[-1] != '？':
            combine_text = '"context":"问题:' + question + '？回答:" "target": "' + answer + '"'
        else:
            combine_text = '"context":"问题:' + question + '回答:" "target": "' + answer + '"'
        result_list.append(combine_text)
    
    with open('dataset/medi/knowledge_base.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_list))


if __name__ == '__main__':
    
    json_to_txt('dataset/medi/merged.json')

