import time
import sys

import jsonlines

sys.path.insert(0, '/media/kou/Data1/htc/FastChat/fastchat/serve')
from tqdm import tqdm
import os
import json
import numpy as np
from random import sample
import re
#
# key = [
# 'asdfa'
# ]
#
# i = 0
# import openai
# openai.api_key = key[i]


def datatrans(scene):
    rooms = scene['rooms']
    roomout = []
    for room in rooms:
        roomout.append(room['roomType'])
    boundingbox = {}
    states = {}
    x = 0
    assetsid = []
    for onething in scene['objects']:
        assetsid.append(onething['assetId'])
    # for onething in random_sample:
    #     state = {}
    #     for onekey,onevalue in onething.items():
    #         if onekey != 'children' and onekey != 'kinematic' and onekey != 'layer' and onekey != 'assetId' and onekey != 'rotation':
    #             if onekey == 'id':
    #                 state['belong to room '] = onevalue[:1]
    #             else:
    #                 state[onekey] = onevalue
    #     states[onething['assetId']] = state
    return roomout,boundingbox,assetsid



def get_completion(messages, model="gpt-3.5-turbo"):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message["content"]

prompts='''
'Analyze the 3D object model from the given caption: ' 
'1. Write a detailed caption by classifying and describing different rooms in 150-200 words, illustrating their types, appearance and other information such as functionalities, usages, daily-life knowledge.' 
'2. Generate 5 single-round Q&As about 5 different object in rooms, considering diverse aspects of the object based on the provided captions and your new captions,like number,usage,material and daily-life knowledge. '
'3. Construct 5 single-round Q&As,focusing on object' coordinate system location and belonged room. ' 
'Format your response as json format:' 
{
    "caption": "description",
    "single conversation": [{"Q": "Q", "A": "A"}*5],
    "detection conversation": [{"Q": "Q", "A": "A"}*5],
}
remember response in a definite and human-like tone,the response is better to be diverse,don't output parts of input,ignore objects' id
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=int, default=0, help="an optional integer argument")
args = parser.parse_args()

if __name__ == "__main__":
    Task_type = "Caption"#VQA,Caption,Relation
    rootpath = "/media/kou/Data1/htc/LAMM/answers/"
    Level = "Scene"#Scene,Object
    if Task_type == "VQA":
        path = rootpath+r'VQA.jsonl'
    elif Task_type == "Caption":
        path = rootpath+r'Caption.jsonl'
    else:
        path = rootpath+r'Relation.jsonl'
    sys_prompt = {
        "VQA":"Your answer must be only a overall score.Use 0 to 10 to give the score. You are the quality evaluator of the generated text content, which is a three round Q&As generated based on the reality of the scenario.You'll be informed about the rooms and objects included in the 3D scene, and your taske is to evaluate the quality of the five rounds QA, including whether they are five rounds of QA, and the quality of the text.",
        "Relation":"Your answer must only be an overall score. Use 0 to 10 to give the score. You are the quality evaluator of the generated text content, which is a relational inference generated based on the reality of the objects in the scene. You will be informed about the rooms and objects included in scene, and your task is to evaluate the accuracy of relational reasoning.",
        "Caption": "Your answer must be only a overall score.Use 0 to 10 to give the score. You are the quality evaluator of the generated text content, which is a long caption generated based on the reality of the scene. You will be informed about the rooms and objects included in the 3D scene, and your task is to evaluate the quality of the subtitles."
    }
    # evaluation = [Task_type]

    result_filepath = rootpath+"/Gpt_results/"+Task_type+"_Eva.json"
    if os.path.exists(result_filepath):
        Eva = open(result_filepath,'r')
        Eva = json.load(Eva)
        evaluatelist = []
        for a, b in enumerate(Eva):
            if int(list(b.keys())[0]) != -1:
                evaluatelist.append(int(list(b.keys())[0]))
        print("Scores : "+str(sum(evaluatelist)/len(evaluatelist)))
        sys.exit()
    else:
        Eva = 40*['None']
        evaluatelist = []

    sence = open(path, 'r')
    sence = jsonlines.Reader(sence)

    from cli import vicuna
    from inference import chat

    answer = ''
    model, tokenizer, chatio = vicuna()

    for index,inf in enumerate(tqdm(sence)):
        #不同模型输出id不同
        # if index not in evaluatelist:
        #     continue
        pred = inf['text']
        id = inf['id']
        if Level == 'Object':
            id = re.sub(r".*\_", "", id)
            #id = re.sub(r".*\_", "", id[8:])
            id = re.sub(r"\d+.*", "", id)
            input = '(1) The object is '+ id+ '(2)Generated text is '+pred
        else:
            file = '(' + str(id) + ').json'
            filepath = os.path.join("/media/kou/Data1/htc/myjson", file)
            if not os.path.exists(filepath):
                continue
            f = open(filepath, 'r')
            GT = json.load(f)
            objpath = r'/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Classification.json'
            obj = open(objpath,'r')
            obj = json.load(obj)
            objects = obj[int(id)]
            rooms,boundingbox,states = datatrans(GT)
            input = "(1)Truth is rooms:"+ str(rooms) +", objects:"+str(objects)+" (2)Generated text is "+pred

        # input = {"role": "user","content":prompt+'\n'+"(1)"+str(rooms)+'\n'+"(2)"+str(boundingbox)+'\n'+"(3)"+str(states)}

        input = sys_prompt[Task_type]+input
        answer = chat(input, model, tokenizer, chatio)
        # 用空格分开所有单词
        words = answer.split()
        # 排除包含 "3D" 的单词
        filtered_words = [word for word in words if '3D' not in word and '(1)' not in word and '1:' not in word and '1.' not in word]
        # 将剩余的单词拼接回去
        answer = ' '.join(filtered_words)
        score = re.findall(r'\d+(?:\.\d+)?', answer)
        if len(score)==0:
            score =[-1]
            print(answer)

#################################GPT#################################
        # sys_message = {"role": "system", "content": sys[Task_type]}
        # messages = [sys_message, {"role": "user", "content": input}]
        # print(messages)
        # while True:
        #     try:
        #         response = get_completion(messages)
        #         # response = json.loads(response, strict=False)
        #         print(str(id)+'成功')
        #         # print(response)
        #     except Exception as e:
        #         print(e)
        #         openai.api_key = key[(i + 1)%1]
        #         time.sleep(30)
        #         continue
        #     else:
        #         break
        #

        Eva[id-460] = {score[0]:answer}
        with open(result_filepath, 'w') as f:
            # 把列表写入到文件里，转换成json格式
            json.dump(Eva, f, indent=4)

