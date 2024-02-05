import openai
from tqdm import tqdm
import pandas as pd
import re
import csv
import json
from sklearn.metrics import recall_score, accuracy_score,f1_score,precision_score
import uuid
import os
import time
import torch
import time

print("Start")
time.sleep(5)  # 暂停5秒
print("End")


device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
DATA_PATH= "../Sohu2022_data/nlp_data"





# 加载数据集
def load_data_1(filename): #处理一个文本具有多个aspect的情况，所有aspect在一条文本后面
    D = []
    D_id= []
    D_entity_plo=[]
    # seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f.readlines(), desc="Loading data"):
            taskData = json.loads(l.strip())
            id=taskData["id"]
            # print(id)
            D_id.append(id)
            text2 = ' The simplified sentence should contain the following words: '
            content=str(taskData['content'])
            entity=list(taskData['entity'].keys())
            entity_plo = taskData['entity']
            result = ', '.join([str(item) for item in entity])
            D.append(content+text2+result)
            D_entity_plo.append(entity_plo)
            # D_entity.append(taskData['entity'])
    return D_id,D,D_entity_plo


os.environ['OPENAI_API_KEY'] = "your APT"
os.environ['OPENAI_API_BASE'] = "XXX"

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.environ['OPENAI_API_BASE']
openai.Model.list()


def query_chat_model(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            # {"role": "system", "content": "The text you'll be dealing with next all comes from Twitter, and you're required to do some sentiment analysis on the sentences."},  #使用系统消息来描述一个特定的场景，比如：“你是一个历史学家，专门研究中世纪的历史。”这样的系统消息可以帮助模型了解它应该扮演的角色，并根据这个角色来生成更加相关和准确的回答。
            # {"role": "system", "content": "Simplify this sentence while simultaneously retaining the meaning and relevant emotional words."},
            # {"role": "system","content": "Strive to ensure that the number of words in the simplified sentences is less than 15."},
            # {"role": "system", "content": "The sentence should not contain special characters."},
            # {"role": "system", "content": "After a sentence exceeds 15 words, it needs to be further condensed, with key emotional terms extracted as necessary to meet the 15-word requirement."},
            {"role": "system", "content": "Simplify the following sentence."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response['choices'][0]['message']['content']
    return reply,response


id,data,entity_plos=load_data_1(DATA_PATH+os.sep+train_file)
count_right=0
# print(data)
count=1
Breakpoint=6241
for i in tqdm(range(len(data[Breakpoint:]))):
    # print(data[i],yy[i])
    # if i%50==0:
    #     time.sleep(5)
    reply,response=query_chat_model(data[i+Breakpoint])
    # print(reply)

    if i+1+Breakpoint==id[i+Breakpoint]:
        with open(DATA_PATH+os.sep+"Twitter_Train_output-gpt.txt","a") as f:
            shuju = {
                "id": id[i+Breakpoint],
                "content":reply.rstrip(),
                "entity": entity_plos[i+Breakpoint]
            }
        # print(shuju)

            json.dump(shuju, f)
            f.write('\n')
        print(count)
        count+=1
    # with open(DATA_PATH+os.sep+"Restaurants_Train_output-gpt.txt", 'a') as f:
    #     # print(reply)
    #     f.write(reply+"\t"+yy[i]+'\n')
#     if reply==yy[i]:
#         count_right+=1
# print(f"正确率={count_right/len(data[:100])}")
