'''
对传统的ABSA数据进行处理，使得同一个句子不同的aspect合成为一个句子，最终生成的样式如：
{"id": 4, "content": "did not enjoy the new windows 8 and touchscreen functions .", "entity": {"windows 8": -1, "touchscreen functions": -1}}
'''
def load_data(tsvpath):
    '''
    适配ABSA的数据格式

    :param tsvpath:
    :return:
    '''

    fin = open(tsvpath, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    data=[ ]
    yy=[ ]
    text1=[]
    id=1

    for i in range(0, len(lines), 3):
        aspect = lines[i + 1].lower().strip()
        text=lines[i].replace("$T$", aspect).lower().strip()
        text=text
        polarity = lines[i + 2].strip()

        shuju= {
            "id":id,
            "content": text.rstrip(),
            "entity": {
                aspect.rstrip(): int(polarity)
            }
        }
        # data.append(shuju)   #一句话多个方面占居多行
        # 处理同一句话多个方面，只保留一句话，方面往后加的目的
        existing_data = next((d for d in data if d["content"] == text), None)
        if existing_data:
            # 如果已存在，将当前的 entity 加在第一个之后
            existing_data["entity"].update(shuju["entity"])
        else:
            # 如果不存在，直接添加到数据列表中

            data.append(shuju)
            id = id + 1
        # data.append(shuju)
    # new_text = [data[i] + " " + text1[i] for i in range(len(data))]
    import json


    return data#data.values[:,1:2].tolist() #data.values[:,2:3]：文本    yy：情感极性

i=1
data=load_data("F:\ExpCode\论文复现\ABSA-PyTorch-master\datasets/semeval14/Laptops_Test_Gold.xml.seg")
import json
with open("F:\ExpCode\论文复现\ABSA-PyTorch-master\datasets/semeval14/Laptops_Test_output-no[cls]-2.0.txt", "w") as file_1:  #1-aspect
    # for item in data:
    #     line = json.dumps(item)
    #     file.write(line + "\n")
     for item in data:
            line = json.dumps(item)
            json_data = json.loads(line)
            value = json_data['entity']  # 使用键来访问值
            # if "0" in str(value):
            #     # i += 1
            #     # if i%3==0:

            file_1.write(line + "\n")


