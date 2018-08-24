# -*- encoding: utf-8 -*-
import json
file_list = ['./data_dev_preprocessed.json', './data_test_preprocessed.json', './data_train_preprocessed.json']
simbol_list = [ '，', '。']
for file_name in file_list:
    current_file = open(file_name+'.bak', 'r', encoding='utf-8')
    new_file = open(file_name, 'w', encoding='utf-8')
    for line in current_file.readlines():
        que_json = json.loads(line)
        buf = que_json['documents'][0]['title'].replace('，','').replace('。', '')
        que_json['documents'][0]['title'] = buf
        buf = que_json['documents'][0]['paragraphs'][0].replace('，','').replace('。', '')
        que_json['documents'][0]['paragraphs'][0] = buf
        buf = que_json['question'].replace('，','').replace('。', '')
        que_json['question'] = buf
        if que_json['answers']:
            buf = que_json['answers'][0].replace('，','').replace('。', '')
            que_json['answers'][0] = buf
            buf = [x for x in que_json['segmented_answers'][0] if x not in simbol_list]
            que_json['segmented_answers'][0] = buf
        buf = [x for x in que_json['documents'][0]['segmented_title'] if x not in simbol_list]
        que_json['documents'][0]['segmented_title'] = buf
        buf = [x for x in que_json['documents'][0]['segmented_paragraphs'][0] if x not in simbol_list]
        que_json['documents'][0]['segmented_paragraphs'][0] = buf
        buf = [x for x in que_json['segmented_question'] if x not in simbol_list]
        que_json['segmented_question'] = buf
        new_line = json.dumps(que_json, ensure_ascii = False)
        new_file.write(new_line+'\n')
    current_file.close()
    new_file.close()
