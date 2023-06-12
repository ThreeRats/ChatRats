import csv
import json

def preprocess(answer_csv, question_csv, json_file):

    '''
    将数据集处理为需要的格式
    '''

    answer_data = {}
    question_data = {}

    with open(answer_csv, 'r', encoding='utf-8') as file1:
        reader1 = csv.DictReader(file1)
        for row in reader1:
            id = row['question_id']
            answer_data[id] = row['content']
    

    with open(question_csv, 'r', encoding='utf-8') as file2:
        reader2 = csv.DictReader(file2)
        for row in reader2:
            id = row['question_id']
            question_data[id] = row['content']
    
    result_list = []

    for id in question_data:
        
        temp = {
            'question':question_data[id],
            'answer':answer_data[id]
        }
        result_list.append(temp)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False)

def split_train_val_test(json_file, val_rate=0.1, test_rate=0.1):

    '''
    按照比例划分训练集，验证集，测试集
    
    json_file:经过预处理的完整数据集
    val_rate:验证集比例
    test_rate:测试集比例
    '''

    json_list = json.load(open(json_file, 'r', encoding='utf-8'))
    train_rate = 1 - val_rate - test_rate

    print(f'数据集一共有{len(json_list)}条数据')

    train_list = json_list[ : int(len(json_list) * train_rate)]
    val_list = json_list[int(len(json_list) * train_rate) : int(len(json_list) * (train_rate + val_rate))]
    test_list = json_list[int(len(json_list) * (train_rate + val_rate)) : ]

    print(f'训练集有{len(train_list)}条数据')
    print(f'验证集有{len(val_list)}条数据')
    print(f'测试集有{len(test_list)}条数据')

    with open('dataset/medi/train_set.json', 'w', encoding='utf-8') as train_file:
        json.dump(train_list, train_file, ensure_ascii=False)
    
    with open('dataset/medi/val_set.json', 'w', encoding='utf-8') as val_file:
        json.dump(val_list, val_file, ensure_ascii=False)
    
    with open('dataset/medi/test_set.json', 'w', encoding='utf-8') as test_file:
        json.dump(test_list, test_file, ensure_ascii=False)


if __name__ == '__main__':

    preprocess('dataset/medi/answer.csv', 'dataset/medi/question.csv', 'dataset/medi/merged.json')
    split_train_val_test('dataset/medi/merged.json')
    