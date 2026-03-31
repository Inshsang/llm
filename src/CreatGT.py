from functools import lru_cache
import sys
import math
import os
import json
from typing import Any, Dict, List
import numpy as np
from random import sample
import random
import re
import math
import open3d as o3d
import multiprocessing
import jsonlines

dirpath = os.path.split(os.path.realpath(__file__))[0]

def getdir(p0,p1):
    x0,y0,z0 = p0[:3]
    x1, y1, z1 = p1[:3]
    dis = 0.5
    if abs(y0-y1)>dis and abs(x0-x1)<dis and abs(z0-z1)<dis:#上下
        if y0 > y1:
            return random.randint(240,269),0
        elif y0 < y1:
            return random.randint(210,239),0
    elif abs(y0-y1)<2*dis and abs(x0-x1)>dis and abs(z0-z1)<dis:#左右
        if x0 > x1:
            return random.randint(60,89),0
        elif x0 < x1:
            return random.randint(30,59),0
    elif abs(y0-y1)<2*dis and abs(x0-x1)<dis and abs(z0-z1)>dis:#前后
        if z0 > z1:
            return random.randint(90,119),0
        elif z0 < z1:
            return random.randint(120,149),0
    elif abs(y0-y1)<2*dis and abs(x0-x1)>dis and abs(z0-z1)>dis:#斜方向
        if z0 > z1 and x0 > x1:
            return random.randint(150,179),0#a在b左前方
        elif z0 > z1 and x0 < x1:
            return random.randint(180,209),0#a在b右前方
        elif z0 < z1 and x0 > x1:
            return random.randint(180,209),1#b在a左后方
        elif z0 < z1 and x0 < x1:
            return random.randint(150,179),1#b在a右后方
    else:
        return 0,-1

def getrandom(x,y):
    return random.randint(x, y)

def getsinglejson(src_id,id,pcl_path,newchat,task_type):
    """
    :param src_id:
    :param id:
    :param pcl_path:
    :param newchat:
    :return:
    """
    singlejson = {}
    singlejson["src_id"] = str(src_id)
    singlejson["id"] = str(id)
    singlejson["pcl"] = pcl_path
    singlejson["conversations"] = newchat
    singlejson["task_type"] = task_type
    singlejson["src_dataset"] = "Mydata"
    return singlejson

def getVGanswer(GT,Answer,Question):

    out = ''
    answer = ''
    for index,list in enumerate(GT):
        name = list['name']
        value = list['BoundingBox']
        x = round(value[0],1)
        y = round(value[1], 1)
        z = round(value[2], 1)
        random_number = random.randint(30, 59)
        random_number = str(random_number)
        temp = Answer[random_number]
        temp = re.sub(r"{C}", "(obj"+str(index)+"):"+name, temp)
        temp = re.sub(r"{P}", str([x,y,z]), temp)
        answer = temp
        random_number = random.randint(0, 29)
        random_number = str(random_number)
        out = Question[random_number]
        out = re.sub(r"{C}", name, out)

    return answer,out

def getcouting(GT,Answer):
    answer = ''
    for list in GT:
        name, num = list
        random_number = random.randint(0, 29)
        random_number = str(random_number)
        temp = Answer[random_number]
        temp = re.sub(r"{C}", name, temp)
        temp = re.sub(r"{N}", str(num), temp)
        answer += temp

    return answer

def getroom(GT,Answer):
    answer = ''
    # N = len(GT)
    # random_number = random.randint(30, 59)
    # random_number = str(random_number)
    # temp = Answer[random_number]
    # answer = re.sub(r"{N}", str(N), temp)
    for list in GT:
        R, P = list
        random_number = random.randint(0, 29)
        random_number = str(random_number)
        temp = Answer[random_number]
        temp = re.sub(r"{R}", R, temp)
        temp = re.sub(r"{P}", str(P), temp)
        answer = answer + temp

    return answer

def chouyang(GT,Answer):
    answer = ''
    _, obj0 = GT[getrandom(0, len(GT) - 1)]
    _, obj1 = GT[getrandom(0, len(GT) - 1)]
    p0 = str(tuple(list(obj0.values())[0]))
    p1 = str(tuple(list(obj1.values())[0]))
    while (obj0 == obj1):
        _,obj1 = GT[getrandom(0, len(GT) - 1)]
    head0 = Answer[str(getrandom(0, 29))]
    head1 = Answer[str(getrandom(0, 29))]
    head0 = re.sub(r"{C}", list(obj0.keys())[0], head0)
    head0 = re.sub(r"{P}", p0, head0)
    head1 = re.sub(r"{C}", list(obj1.keys())[0], head1)
    head1 = re.sub(r"{P}", p1, head1)
    answer = head0 + head1
    return answer,obj0,obj1

def chouyang_train(simgt,Answer):
    class_list = list(simgt.keys())
    class_num = random.sample(range(0,len(class_list)),2)
    answer = ''
    for a,b in zip(class_num[0::2],class_num[1::2]):
        class_a = class_list[a]
        class_b = class_list[b]
        pos_a = simgt[class_a]['BoundingBox']
        pos_b = simgt[class_b]['BoundingBox']

        direaction, flag = getdir(pos_a, pos_b)

        if flag == -1:
            continue
        if flag == 1:
            temp = class_a
            class_a = class_b
            class_b = temp
        dir = Answer[str(direaction)]
        answer = re.sub(r"C1", class_a, dir)
        answer = re.sub(r"C2", class_b, answer)

    return answer,direaction,class_a,class_b,flag

def chouyang_test(class_list,position,Answer):

    class_num = random.sample(range(0,len(class_list)),2)
    answer = ''
    for a,b in zip(class_num[0::2],class_num[1::2]):
        class_a = class_list[a]
        class_b = class_list[b]
        pos_a = position[class_a]
        pos_b = position[class_b]


        direaction, flag = getdir(pos_a, pos_b)

        if flag == -1:
            continue
        if flag == 1:
            temp = class_a
            class_a = class_b
            class_b = temp
        dir = Answer[str(direaction)]
        answer = re.sub(r"C1", class_a, dir)
        answer = re.sub(r"C2", class_b, answer)

    return answer,direaction,class_a,class_b,flag

def get_answer(Answer,flag,direaction,class_a,class_b):
    false = []
    if direaction>=30 and direaction<=149:      #前后左右
        if direaction>=30 and direaction<=89:
            fa = Answer[str(random.randint(90, 119))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(120, 149))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
        else:
            fa = Answer[str(random.randint(30, 59))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(60, 89))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)

    if direaction>=150 and direaction<=209:      #斜向

        if direaction>=150 and direaction<=179:
            fa = Answer[str(random.randint(180, 209))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(180, 209))]
            fa = re.sub(r"C1", class_a, fa)
            fa = re.sub(r"C2", class_b, fa)
            false.append(fa)
        else:
            fa = Answer[str(random.randint(150, 179))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(150, 179))]
            fa = re.sub(r"C1", class_a, fa)
            fa = re.sub(r"C2", class_b, fa)
            false.append(fa)
    fa = Answer[str(direaction)]  # 反向答案
    fa = re.sub(r"C1", class_b, fa)
    fa = re.sub(r"C2", class_a, fa)
    false.append(fa)
    random.shuffle(false)
    return false

def unique_name_mask(lst):
    name_count = {}  # 用于记录每个name出现的次数

    # 统计每个name的出现次数
    for item in lst:
        name = item.get('name', '')  # 获取字典中的name值
        if name:
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1

    # 创建布尔类型的掩码列表，标记唯一存在的name元素
    mask = []
    for item in lst:
        name = item.get('name', '')
        if name:
            if name_count[name] == 1:
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(False)  # 如果字典中没有name键，则默认为False

    return mask


def get_testPosition(answer,false_list,count,template, class_a, class_b,id):
    result = {}
    result["question_id"] = count

    result["pcl"] = template['pcl'][:-7]+str(id)+'.npy'

    result["id"] = str(id)
    result["src_dataset"] = "Mydata"

    answer_pos = random.randint(0,3)
    result["gt_choice"] = answer_pos

    false_list.insert(answer_pos, answer)
    gt_choices = false_list
    result["gt_choices"] = false_list

    answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) "}
    result["sentences"] = answer_query[str(answer_pos)] + answer

    query = "What is the positional relationship of "+class_a+" and "+class_b+"?"+" \n Options: "
    for index,i in enumerate(gt_choices):
        query = query + answer_query[str(index)] + str(i)
    result["query"] = query

    return result

def getPosrealtion(GT,Answer):#GT,pos,Answer
    answer,direaction,class_a,class_b,flag = chouyang_train(GT,Answer)
    # answer, obj0, obj1,_,_ = chouyang_test(GT, pos,Answer)
    # direaction,flag = getdir(tuple(list(obj0.values())[0]),tuple(list(obj1.values())[0]))
    num = 0
    while(flag == -1):#重来
        #print("重新抽样")
        if num>=100000:
            return 0
        num += 1
        answer,direaction,class_a,class_b,flag = chouyang_train(GT,Answer)

    return answer,class_a,class_b

def Train_PositionRelation():
    outjson = []
    Q, A, GT, result = filepath('PositionRelation')
    Question, Answer, GT = loading(Q, A, GT)

    for id in range(0, 460):
        GTid = str(id)
        simpleGT = GT[str(id)]
        for x in range(4):
            answer,class_a,class_b = getPosrealtion(simpleGT, Answer)
            random_number = random.randint(0, 29)
            random_number = str(random_number)
            question = Question[random_number]
            question = re.sub("{C1}", class_a, question)
            question = re.sub("{C2}", class_b, question)
            pcl_path = "scene/"+GTid+".npy"
            conversations = [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]
            singlejson = getsinglejson(str(id), str(x), pcl_path, conversations, "PositionRelation")

            outjson.append(singlejson)

        print(id,"OK")

    return result, outjson

def get_testClass(gt,num):
    result = {}
    src_id = re.sub(r"_.*","",gt)

    classname = re.sub(r".*\d_", "", gt)
    classname = re.sub(r"\d.*", "", classname)
    result["question_id"] = num
    result["pcl"] = "objects/"+gt+".npy"

    result["id"] = src_id
    result["src_dataset"] = "Mydata"

    path = "/data/HTC/Data/dataset/object_add/my_names.json"
    ALL = json.load(open(path,'r'))
    random_number = random.sample(range(0,len(ALL)),5)
    random_class = [ALL[i] for i in random_number]
    while (classname in random_class):
        random_number = random.sample(range(0, len(ALL)), 5)
        random_class = [ALL[i] for i in random_number]

    answer = random.randint(0,5)        #答案序号
    result["gt_choice"] = answer

    gt_choices = random_class.insert(answer, classname)#答案列表
    gt_choices = random_class
    result["gt_choices"] = gt_choices
    answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) ","4":" (E) ","5":" (F) "}
    # result["sentences"] = answer_query[str(answer)] + classname

    # Q = root + "/BenchMark/Task/Template/Q_Classification.json"
    # q = open(Q, 'r')
    # Question = json.load(q)
    # Question = Question[str(random.randint(0,29))]
    Question = "What's the 3D point cloud about?"
    query = Question+" \n Options: "
    for index,i in enumerate(gt_choices):
        query = query + answer_query[str(index)] + i
    result["query"] = query
    result["sentences"] = answer_query[str(answer)]+classname
    # result["query"] = Question
    # result["sentences"] = classname
    return result

def get_trainClass(name):
    classnum = re.sub(r"_.*","",name)
    classname = re.sub(r".*\d_", "", name)
    classname = re.sub(r"\d.*", "", classname)

    Q = "/data/HTC/Data/dataset/Benchmark/Task/Template/Q_Classification.json"
    q = open(Q, 'r')
    Question = json.load(q)
    Question = Question[str(random.randint(0, 29))]
    A = "/data/HTC/Data/dataset/Benchmark/Task/Template/A_Classification.json"
    a = open(A, 'r')
    Answer = json.load(a)
    Answer = Answer[str(random.randint(0, 29))]
    Answer = re.sub("{C}", classname, Answer)

    #物体对齐数据集
    Answer = classname

    conversation = [{"from": "human", "value": Question},{"from": "gpt","value": Answer}]
    pcl_path = "Objects/" + name+".npy"

    single = getsinglejson(classnum, classnum, pcl_path, conversation,"Classification3d")

    return single

def get_testCounting(gtnum,index,objclass,num):
    result = {}
    result["question_id"] = index + 1
    result["pcl"] = "scene/"+str(gtnum)+'.npy'


    result["id"] = str(gtnum)
    result["src_dataset"] = "Mydata"

    path = "G:\event\htc\MYDATA\BenchMark\Task\GT\Counting.json"

    random_number = random.sample(range(1,10),5)

    while (num in random_number):
        random_number = random.sample(range(1, 10), 5)

    answer_pos = random.randint(0,5)
    result["gt_choice"] = answer_pos

    random_number.insert(answer_pos, num)
    gt_choices = random_number
    result["gt_choices"] = gt_choices

    answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) ","4":" (E) ","5":" (F) "}
    result["sentences"] = answer_query[str(answer_pos)] + str(num)

    query = "How many "+objclass+" are in the scene?"+" \n Options: "
    for index,i in enumerate(gt_choices):
        query = query + answer_query[str(index)] + str(i)
    result["query"] = query

    return result

def point2box(points):
    x_max = 0
    z_max = 0
    x_min = 100
    z_min = 100

    for i in points:
        if x_max < i['x']:
            x_max = i['x']
        if z_max < i['z']:
            z_max = i['z']
        if x_min > i['x']:
            x_min = i['x']
        if z_min > i['z']:
            z_min = i['z']
    y_mid = round(points[0]['y'],3)
    y_mid = y_mid/2
    h = y_mid*2

    x_mid = round((x_min + x_max)/2,3)
    z_mid = round((z_min + z_max) / 2,3)

    l = round(x_max - x_min,3)
    w = round(z_max - z_min,3)
    answer = [x_mid,z_mid,y_mid,l,w,h]
    return answer

def filepath(Class):
    root = "/data/HTC/Data/dataset"
    # root = "/media/cvlab/Data/htc/MYDATA"
    Q = root + "/Benchmark/Task/Template/Q_"+Class+".json"
    A = root + "/Benchmark/Task/Template/A_"+Class+".json"
    GT = root + "/Benchmark/Task/GT/"+Class+".json"

    if Class == "Detection":
        A = root + "/Benchmark/Task/Template/A_" + "VisualGrounding" + ".json"
        GT = root + "/Benchmark/Task/GT/" + "Detection" + ".json"

    if Class == "VisualGrounding":
        GT = root + "/Benchmark/Task/GT/"+"VisualGrounding"+".json"
    if Class == "PositionRelation":
        GT = root + "/Benchmark/Task/GT/"+"VisualGrounding"+".json"

    result = "/data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/Temp.json"
    return Q,A,GT,result

def loading(Q,A,GT):
    q = open(Q, 'r')
    Question = json.load(q)
    a = open(A, 'r')
    Answer = json.load(a)
    gt = open(GT, 'r')
    GT = json.load(gt)
    return Question,Answer,GT

# ## SCaption and SVQA
# for one in GT:
#     one['query'] = "Generate 5 round Q&A conversation of the given point cloud."
#     # one['sentences'] = one['sentences'][5:]
#     outjson.append(one)

class_mapping = {
    "cabinet": 0,
    "bed": 1,
    "chair": 2,
    "sofa": 3,
    "diningtable": 4,
    "doorway": 5,
    "window": 6,
    "shelf": 7,
    "painting": 8,
    "countertop": 9,
    "desk": 10,
    # "curtain": 11,  #
    "fridge": 12,
    # "showercurtrain": 13,  #
    "toilet": 14,
    "sink": 15,
    # "bathtub": 16,  #
    "garbagecan": 17,
}

def getanswerDe(oneGT,Answer):
    answer = ''
    new_gt = []
    for i in oneGT:
        if not len(i):
            continue
        name, bbox = i['name'], i['BoundingBox']
        if name.lower() not in class_mapping.keys():
            continue
        new_gt.append(i)
    oneGT = new_gt

    if len(oneGT)>=10:
        class_num = random.randint(2, 10)
        random_list = random.sample(range(0, len(oneGT)), class_num)
        oneGT = [oneGT[i] for i in random_list]

    i = 0
    box = []
    for one in oneGT:
        name, bbox = one['name'],one['BoundingBox']
        answer += f"(obj{str(i)}):{name.lower()}! "
        box.append(bbox)
        i += 1
        # obj = random.randint(0,29)
        # obj = Answer[str(obj)]
        # singleans = re.sub("{C}", name, obj)
        # singleans = re.sub("{P}", str(bbox), singleans)
        # answer += singleans
    return answer,box

def getanswerRoomDe(oneGT,Answer):
    answer = ''
    for name,bbox in oneGT.items():
        bbox = point2box(bbox)
        obj = random.randint(0,29)
        obj = Answer[str(obj)]
        singleans = re.sub("{R}", name, obj)
        singleans = re.sub("{P}", str(bbox), singleans)
        answer += singleans
    return answer



# Test Navigation  ##
def Test_Navigation():
    countnum = 0
    outjson = []
    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Temp.json"
    GT_path = r"/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Navigation.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Question_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Navigation.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)

    for key,value in GT.items():

        if int(key) <= 459 or int(key) >= 500:
            continue
        # oneGT = list(value.values())
        for one in value:
            for name,positions in one.items():
                singlejson = {}
                countnum += 1
                singlejson["positions"] = positions

                ans_num = random.randint(0,29)
                que = Question[str(ans_num)]
                que = re.sub(r"{C}", name, que)
                que = re.sub(r"{P}", str(positions[0]), que)
                singlejson["question"] = que
                singlejson["query"] = que
                singlejson["question_id"] = countnum
                pcl_path = "scene/" + str(key) + '.npy'
                singlejson["pcl"] = pcl_path
                singlejson["id"] = str(key)
                singlejson["src_dataset"] = "Mydata"
                outjson.append(singlejson)

            print(key,"OK")
    return result, outjson

## Test RoomDetection  ##
def Test_RoomDetection():
    outjson = []
    result = "G:\event\htc\MYDATA\BenchMark\Task\Task_Reconstruct\Temp.json"
    countnum = 0
    GT_path = "G:\event\htc\MYDATA\BenchMark\Task\GT\RoomDetection.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Question_path = "/MYDATA/BenchMark/Task/Template_v0\Q_RoomDetection.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)

    for key,value in GT.items():

        if int(key) <= 459 or int(key) >= 500:
            continue
        # oneGT = list(value.values())
        object = []
        for label,one in value.items():
            one = point2box(one)
            object.append({'label':label,'bbox':one})
        singlejson = {}
        countnum += 1
        # answer =one['BoundingBox']
        singlejson["object"] = object

        ans_num = random.randint(30,59)
        que = Question[str(ans_num)]
        # que = re.sub(r"{C}", one["name"], que)
        singlejson["question"] = que
        singlejson["query"] = que + " Please locate rooms' position with the coordinate of center x, y, z and its length, width and height,represented as (x,y,z,l,w,h)"
        singlejson["question_id"] = countnum
        pcl_path = "scene/" + str(key) + '.npy'
        singlejson["pcl"] = pcl_path
        singlejson["id"] = str(key)
        singlejson["src_dataset"] = "Mydata"
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson
#
# Test VisualGrounding  ##
def Test_VisualGrounding():
    countnum = 0
    outjson = []
    result = "G:\event\htc\MYDATA\BenchMark\Task\Task_Reconstruct\Temp.json"
    Onlyobj = r"G:\event\htc\MYDATA\BenchMark\Task\GT\Counting.json"
    oo = open(Onlyobj,"r")
    Only = json.load(oo)
    GT_path = "G:\event\htc\MYDATA\BenchMark\Task\GT\VisualGrounding.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Question_path = "/MYDATA/BenchMark/Task/Template_v0\Q_VisualGrounding.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)

    for key,value in GT.items():
        if int(key) <= 459 or int(key) >= 500:
            continue
        oneGT = list(value.values())
        for one in oneGT:
            if one["name"].lower() in [name for name,num in Only[int(key)].items() if num == 1]:
                pass
            else:
                continue
            countnum += 1
            singlejson = {}
            answer =one['BoundingBox']
            singlejson["object"] = answer

            ans_num = random.randint(0,29)
            que = Question[str(ans_num)]
            que = re.sub(r"{C}", one["name"], que)
            singlejson["question"] = que
            singlejson["query"] = que + " Please locate its position with the coordinate of center x, y, z and its length, width and height."
            singlejson["question_id"] = countnum
            pcl_path = "scene/" + str(key) + '.npy'
            singlejson["pcl"] = pcl_path
            singlejson["id"] = str(key)
            singlejson["src_dataset"] = "Mydata"
            outjson.append(singlejson)

        print(key,"OK")
    return result, outjson
#



# Train RoomDetection  ##
def Train_RoomDetection():
    outjson = []
    Q, A, GT, result = filepath('RoomDetection')
    Question, Answer, GT = loading(Q, A, GT)
    for key,value in GT.items():
        singlejson = {}
        if int(key) > 459:
            continue

        answer = getanswerRoomDe(value,Answer)
        ans_num = random.randint(30, 59)
        question = Question[str(ans_num)]

        conversations = [{"from": "human","value":question},{"from": "gpt","value":answer}]
        pcl_path = "scene/" + str(key) + '.npy'

        singlejson = getsinglejson(str(key), str(key), pcl_path, conversations, "RoomDetection3d")
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson
## Test Detection  ##
def Multi_Classification():
    countnum = 0
    result = "/media/kou/Data1/htc/MYDATA/Task/Task_Reconstruct/Temp.json"
    outjson = []
    GT_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"
    GT = open(GT_path,'r')
    Question_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Detection.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)
    # for oneline in jsonlines.Reader(GT):
    #     for key1, value1 in oneline.items():
    #         key, value = key1,value1
    #     singlejson = {}
    #     if int(key) >= 500 or int(key)<460:
    #         continue
    #     countnum += 1
    #     obj = []
    #     for one in value:
    #         name = one["name"]
    #         box = one["BoundingBox"]
    #         if name.lower() not in class_mapping.keys():
    #             continue
    #         obj.append({'name':name.lower(),'BoundingBox':box})
    pre_Box = json.load(open("/media/kou/Data1/htc/LAMM/data/metadata/Detection.json"))
    for key1, value1 in pre_Box.items():
        if int(key1) < 460:
            continue
        countnum += 1
        obj = []
        for one in value1:
            name = one["name"]
            box = one["BoundingBox"]
            if name.lower() not in class_mapping.keys():
                continue
            obj.append({'name':name.lower(),'BoundingBox':box})

        singlejson = {}
        que_num = random.randint(0,29)
        question = Question[str(que_num)]+" With obj0:name0, obj1:name1... form of answers."
        singlejson['question_id'] = countnum
        class_num = random.randint(2,5)
        random_list = random.sample(range(0, len(obj)), class_num)
        obj = [obj[i] for i in random_list]
        singlejson['object'] = obj
        singlejson["query"] = question
        pcl_path = "scene/" + str(key1) + '.npy'
        singlejson["pcl"] = pcl_path
        singlejson["id"] = str(key1)
        singlejson["src_dataset"] = "Mydata"
        outjson.append(singlejson)

        print(key1,"OK")
    return result, outjson

def Test_Detection():
    countnum = 0
    result = "/media/kou/Data1/htc/MYDATA/Task/Task_Reconstruct/Temp.json"
    outjson = []
    GT_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"
    GT = open(GT_path,'r')
    Question_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Detection.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)
    for oneline in jsonlines.Reader(GT):
        for key1, value1 in oneline.items():
            key, value = key1,value1
        singlejson = {}
        if int(key) >= 500 or int(key)<460:
            continue
        countnum += 1
        obj = []
        for one in value:
            name = one["name"]
            box = one["BoundingBox"]
            if name.lower() not in class_mapping.keys():
                continue
            obj.append({'name':name.lower(),'BoundingBox':box})

        que_num = random.randint(0,29)
        question = Question[str(que_num)]+" With obj0:name0, obj1:name1... form of answers."
        singlejson['question_id'] = countnum
        singlejson['object'] = obj
        singlejson["query"] = question
        pcl_path = "scene/" + str(key) + '.npy'
        singlejson["pcl"] = pcl_path
        singlejson["id"] = str(key)
        singlejson["src_dataset"] = "Mydata"
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson

# ## Test Pos Relation ##
def Test_PositionRelation():
    Count = 1
    outjson = []
    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Temp.json"
    root = "/media/kou/Data1/htc/MYDATA"
    GT_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Counting.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Answer_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/A_PositionRelation.json"
    Answer = open(Answer_path, 'r')
    Answer = json.load(Answer)
    for id in range(460, 500):

        Testpos = root + "/BenchMark/Task/Test/RoomDetection.json"
        Tp = open(Testpos, 'r')
        Tp = json.load(Tp)
        if id in [int(tp['id'] )for tp in Tp]:
            pass
        else:
            continue
        template = Tp[0]

        pos = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Relationship.json" #所有存在的物体
        pos = open(pos, 'r')
        pos = json.load(pos)

        position = {}

        class_list = [key for key, value in GT[id].items() if value == 1]   #所有数量为1的物体
        classlist = []
        pos = pos[str(id)]
        pos = list(pos.values())
        for values in pos:
            if sum(1 for name in pos if list(name.keys())[0] == list(values.keys())[0]) == 1:     #判断源文件json文件中物体数量是否大于1
                pass
            else:
                continue
            for x,y in values.items():
                x = x.lower()
            if x in class_list:
                values = {x:y}
                classlist.append(x)
                position.update(values)

        for x in range(5):
            answer,direaction,class_a,class_b ,flag= chouyang_test(classlist,position,Answer)

            if (answer == 0 or (direaction>=210 and direaction<=269) or direaction == 0) :
                print("跳过",x,id)
                continue

            false_list = get_answer(Answer, flag, direaction, class_a, class_b)
            if len(false_list)==1:
                print(direaction,false_list)

            singlejson = get_testPosition(answer,false_list,Count,template, class_a, class_b,id)

            outjson.append(singlejson)
            Count += 1

        print(id,"OK")
    return result, outjson
#
# Test Counting ##
def Test_Counting():
    Q_num = 0
    outjson = []
    Q, A, GT, result = filepath('Counting')
    Question, Answer, GT = loading(Q, A, GT)
    for i, single in enumerate(GT):
        if i<460:
            continue
        gt = single
        """
        抽取相应数量的物体，数量==value
        对数量为value的物体抽样
        """
        # single = {key: value for key, value in single.items() if value == 3}
        if len(single)>=8:
            random_number = random.sample(range(0, len(single)), 2)
        # elif len(single)>=4:
        #     random_number = random.sample(range(0, len(single)), 4)
        elif len(single)>=3:
            random_number = random.sample(range(0, len(single)), 3)
        elif len(single)>=2:
            random_number = random.sample(range(0, len(single)), 2)
        elif len(single)>=1:
            random_number = random.sample(range(0, len(single)), 1)
        else:
            continue
        random_class = [list(single)[x] for x in random_number]
        for objclass in random_class:
            single_out = get_testCounting(i, Q_num, objclass, gt[objclass])
            outjson.append(single_out)
            Q_num += 1
            print(i)
    return result, outjson

## Train Classification ##
def Train_Classification():

    result = " "
    outjson = []
    All = json.load(open("/data/HTC/Data/dataset/object_add/my_train.json"))
    for path in All[:20000]:
        single_out = get_trainClass(path)
        
        # #训练集增加选项classification
        # single_test_out = get_testClass(path,0)
        # single_out['conversations'][0]['value'] = single_test_out['query']

        outjson.append(single_out)

    return result,outjson

## Test_Classification ##
def Test_Classification():
    outjson = []
    result = " "
    num = 0
    All = json.load(open("/data/HTC/Data/dataset/object_add/my_test.json"))
    for path in All:
        num += 1
        single_out = get_testClass(path,num)
        outjson.append(single_out)
    return result, outjson

#Train Navigation
def Train_Navigation():
    outjson = []
    Q, A, GT, result = filepath('Navigation')
    Question, Answer, GT = loading(Q, A, GT)

    for id in range(0, 460):
        GTid = str(id)
        if not GTid in list(GT.keys()):
            continue
        ALLpath = GT[GTid]
        for index,path in enumerate(ALLpath):
            singlejson = getNV(path,GTid,index,Answer,Question)
            # if singlejson == 0:
            #     print("跳过",id)
            #     continue
            outjson.append(singlejson)
        print(id,"OK")

    return result, outjson

def getNV(path,GTid,index,Answer,Question):
    random_number = random.randint(0, 29)
    random_number = str(random_number)
    Answer = Answer[random_number]
    Answer = re.sub("{P}", str(list(path.values())[0]), Answer)
    random_number = random.randint(0, 29)
    random_number = str(random_number)
    Question = Question[random_number]
    Question = re.sub("{C}", str(list(path.keys())[0]), Question)
    Question = re.sub("{P}", str(list(path.values())[0][0]), Question)

    pcl_path = "scene/" + str(GTid) + ".npy"
    converstions = [{"from": "human", "value": Question}, {"from": "gpt", "value": Answer}]
    singlejson = getsinglejson(str(GTid), str(index), pcl_path, converstions, "Navigation")

    return singlejson

""" 
Counting的训练数据
"""
def Train_Counting():
    Q, A, GT, result = filepath('Counting')
    Question, Answer, GT = loading(Q, A, GT)
    outjson = []

    for id in range(0, 460):
        GTid = str(id)
        newchat = []

        oneGT = list(GT[id].items())
        # round = math.floor(len(oneGT)/1)+1
        round = len(oneGT)
        for i in range(round):
            answer = getcouting(oneGT[i :(i + 1) ], Answer)
            # answer = getcouting(oneGT[i*5:(i+1)*5], Answer)
            # answer = re.sub(r"{C}", name, Answer[random_number])

            random_number = random.randint(0, 29)
            random_number = str(random_number)
            question = Question[random_number]
            name ,_ = oneGT[i :(i + 1) ][0]
            question = re.sub(r"{C}", name, question)
            newchat = [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]

            pcl_path = "scene/" + str(id) + '.npy'
            singlejson = getsinglejson(str(id),str(i),pcl_path,newchat,"Counting")

            outjson.append(singlejson)
        print(id,"OK")
    return result, outjson[0:17189:3]


# Train Detection // 只保留前25个物体
def Train_Detection():
    choosen_scene = json.load(open("/media/kou/Data1/htc/LAMM/data/meta_file/choosenscene.json"))
    outjson = []
    Q, A, GT, result = filepath('Detection')
    Question, Answer, _ = loading(Q, A, Q)
    with open(GT, 'rb') as f:
        for item in jsonlines.Reader(f):
            if int(list(item.keys())[0])< 500 or int(list(item.keys())[0])>= 10000:
                continue
            # if list(item.keys())[0] not in choosen_scene:
            #     continue
            GTid = list(item.keys())[0]
            oneGT = list(item.values())[0]
            # oneGT = list(oneGT.items())

            que_num = random.randint(0,29)
            question = Question[str(que_num)]+" With obj0:name0, obj1:name1... form of answers. "
            answer,box = getanswerDe(oneGT,Answer)
            if '20' in answer:
                continue
            conversations = [{"from": "human","value": question},{"from": "gpt","value":answer}]
            pcl_path = "scene/" + GTid + '.npy'

            singlejson = getsinglejson(str(GTid),str(GTid),pcl_path,conversations,"Detection3d")
            singlejson["conversations"] = conversations
            singlejson["box"] = box

            outjson.append(singlejson)
            print(GTid,"OK")

    return result, outjson

## *VisualGrounding* ##
"""
每个场景取两个物体组成3W*2个对话
剔除部分场景，空场景和物体不够
"""
def Train_VisualGrounding():
    outjson = []
    countnum = 0
    Q, A, GT, result = filepath('VisualGrounding')
    Question, Answer, _ = loading(Q, A, Q)
    choosen_scene = json.load(open("/media/kou/Data1/htc/LAMM/data/meta_file/choosenscene.json"))

    new_classdict={}
    Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
    for i in Detection:
        key = list(i.keys())[0]
        flag = 0
        if key == '26230' or key == '8213' or (int(key) <= 499) or (key not in choosen_scene):
            continue
        if int(list(i.keys())[0]) < 460:
            continue
        for new_class in i[key]:
            if len(new_class)==0:
                flag = 1
        if flag==1:
            continue
        new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]
        new_classdict[key] = new_classlist

    GT = json.load(open(GT, 'r'))
    for key,value in GT.items():
        GTid = key
        oneGT = value
        oneGT = [value for name,value in oneGT.items() if name.lower() in Detection_class]
        if GTid == '26230' or GTid == '8213' or (int(GTid) <= 499) or (GTid not in choosen_scene):
            continue
        if len(oneGT)==0:        #忽略没有物体的场景
            continue
        if len(oneGT)<2:
            continue
        else:
            num = random.sample(range(len(oneGT)), 2)
        oneGT = [oneGT[i] for i in num]
        if not new_classdict.get(key):
            continue
        obj_num = list(new_classdict[key])
        obj_name = [i['name'] for i in obj_num]
        for i in range(len(oneGT)):
            singlejson = {}
            # singlejson["question_id"] = countnum
            countnum += 1
            answer,question0 = getVGanswer(oneGT[i :(i + 1)], Answer,Question)
            name = oneGT[i :(i + 1)][0]['name']
            if name in obj_name:
                index = obj_name.index(name)
            else:
                continue

            if index>=20:
                continue
            answer = "It is (obj"+str(index) + ") at "+str([round(oneGT[i :(i + 1)][0]["BoundingBox"][0],1),round(oneGT[i :(i + 1)][0]["BoundingBox"][1],1),round(oneGT[i :(i + 1)][0]["BoundingBox"][2],1)])+"."

            conversations = [{"from": "human", "value": question0}, {"from": "gpt", "value": answer}]
            if len(answer)==0:
                continue

            pcl_path = "scene/" + GTid + '.npy'

            singlejson = getsinglejson(str(GTid), str(i), pcl_path, conversations, "VisualGrounding3d")

            outjson.append(singlejson)
        print(GTid,"OK")
    return result, outjson

def crop_point_cloud(point_cloud, bounding_box):
    """
    根据给定的bounding box切割点云场景的一部分。

    Args:
        point_cloud (open3d.geometry.PointCloud): 已加载的点云对象。
        bounding_box (list): 包含[x, y, z, l, w, h]形式的bounding box参数。
            其中 (x, y, z) 是边界框的中心坐标，l 是边界框的长度，w 是宽度，h 是高度。

    Returns:
        open3d.geometry.PointCloud: 切割出的点云对象，如果边界框内没有点云，则返回 None。
    """
    # 解析bounding box参数
    x, y, z, l, w, h = bounding_box

    # 计算边界框的范围
    min_bound = np.array([x - l/2, y - w/2, z - h/2])
    max_bound = np.array([x + l/2, y + w/2, z + h/2])

    # 获取点云数据
    points = np.asarray(point_cloud.points)

    # 筛选出在边界框内的点
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    cropped_points = points[mask]

    if len(cropped_points) == 0:
        return None
    else:
        # 创建新的点云对象并返回
        cropped_point_cloud = o3d.geometry.PointCloud()
        cropped_point_cloud.points = o3d.utility.Vector3dVector(cropped_points)
        return cropped_point_cloud

def check_point_cloud_in_boxes(point_cloud, bounding_boxes):
    """
    根据给定的 bounding boxes 判断对应位置是否存在点云。

    Args:
        point_cloud (open3d.geometry.PointCloud): 已加载的点云对象。
        bounding_boxes (list): 包含多个 [x, y, z, l, w, h] 形式的 bounding box 列表。

    Returns:
        list: 包含布尔值的 mask 列表，每个元素表示对应位置是否存在点云。
    """
    # 获取点云数据
    points = np.asarray(point_cloud.points)

    # 初始化 mask 列表
    mask = []

    # 提取所有 bounding box 的参数
    bbox_params = np.array([bbox['BoundingBox'] for bbox in bounding_boxes])

    # 计算所有 bounding box 的范围
    min_bounds = bbox_params[:, :3] - bbox_params[:, 3:] / 2
    max_bounds = bbox_params[:, :3] + bbox_params[:, 3:] / 2

    # 判断 bounding box 区域内是否存在点云
    for min_bound, max_bound in zip(min_bounds, max_bounds):
        indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        has_points = np.any(indices)
        mask.append(has_points)

    return mask





"""
进一步处理Detection,删除场景中不存在的物体
"""

#检查drc文件转存ply失败的文件
def check(i):
    try:
        # print(i)
        scene_path = "/media/kou/Data3/htc/scene/" + str(i) + ".ply"
        scene = o3d.io.read_point_cloud(scene_path)
        # print(i)
    except Exception  as e:
        print(i)
        print(e)


def process_item(i):
    if i%1000 == 0:
        print(str(i), 'over')
    # 删除场景中不存在的物体
    sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/Detection.json", 'r')
    try:
        for item in jsonlines.Reader(sen):
            if str(i) == list(item.keys())[0]:
                # print(i,list(item.keys())[0],'over')
                scene_path = "/media/kou/Data3/htc/scene/" + str(i) + ".ply"
                scene = o3d.io.read_point_cloud(scene_path)
                items = []
                for one in item[str(i)]:
                    if one:
                       items.append(one)
                if len(items):
                    mask = check_point_cloud_in_boxes(scene,items)
                    pro_item = [item for item, m in zip(item[str(i)], mask) if m]
                else:
                    pro_item = item
                pro_sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/pro_Detection.json", 'a')
                pro_sen.write(json.dumps({str(i):pro_item}) + "\n")
                pro_sen.flush()
                return
    except Exception as e:
        print(e,"bug in ",str(i))

# 给detection排序
def sort_item(i):

    sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/pro_Detection.json", 'r')

    for item in jsonlines.Reader(sen):
        if str(i) != list(item.keys())[0]:
            continue
        else:
            pro_item = item[str(i)]

            pro_sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/PRO_Detection.json", 'a')
            pro_sen.write(json.dumps({str(i):pro_item}) + "\n")
            pro_sen.flush()

            return



# pro_sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/Detection.json", 'a')
#
# for i in range(0,30):
#     scene_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection" + str(i) + ".json"
#     sce = open(scene_path,"r")
#     for item in jsonlines.Reader(sce):
#         pro_sen.write(json.dumps(item) + "\n")

# for i in range(0,30000):
#     if str(i) not in exsiting:
#         print(i)

#
# exsiting = []
# for i in range(0,30):
#     scene_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection" +".json"
#     sce = open(scene_path,"r")
#     for item in jsonlines.Reader(sce):
#         exsiting.append(list(item.keys())[0])
"""
预先定义
"""
Class_ALL = [
    "alarmclock",
    "apple",
    "armchair",
    "baseballbat",
    "basketball",
    "bed",
    "book",
    "boots",
    "bottle",
    "bowl",
    "box",
    "bread",
    "butterknife",
    "candle",
    "cart",
    "cellphone",
    "chair",
    "cloth",
    "clothesdryer",
    "coffeemachine",
    "coffeetable",
    "countertop",
    "creditcard",
    "cup",
    "desk",
    "desklamp",
    "desktop",
    "diningtable",
    "dishsponge",
    "dogbed",
    "doorway",
    "dresser",
    "dumbbell",
    "egg",
    "faucet",
    "floorlamp",
    "fork",
    "fridge",
    "garbagebag",
    "garbagecan",
    "houseplant",
    "kettle",
    "keychain",
    "knife",
    "ladle",
    "laptop",
    "laundryhamper",
    "lettuce",
    "microwave",
    "mug",
    "newspaper",
    "ottoman",
    "painting",
    "pan",
    "papertowelroll",
    "pen",
    "pencil",
    "peppershaker",
    "pillow",
    "plate",
    "plunger",
    "pot",
    "potato",
    "remotecontrol",
    "safe",
    "saltshaker",
    "shelvingunit",
    "sidetable",
    "sink",
    "soapbar",
    "soapbottle",
    "sofa",
    "spatula",
    "spoon",
    "spraybottle",
    "statue",
    "stool",
    "tabletopdecor",
    "teddybear",
    "television",
    "tennisracket",
    "tissuebox",
    "toaster",
    "toilet",
    "toiletpaper",
    "tomato",
    "tvstand",
    "vacuumcleaner",
    "vase",
    "washingmachine",
    "watch",
    "window",
    "winebottle"
]

Detection_class = ["cabinet","bed","chair","sofa","diningtable","doorway","window","shelf", "painting","countertop","desk","fridge","toilet","sink","garbagecan"]

# ## Scene GPT Test:SVQA,Relation,SCaption  ##
# import jsonlines
#
# outjson = []
# for i in range(460,500):
#     gt={}
#
#     # SCaption
#     # gt['query'] = "Write a detailed caption by classifying and describing different rooms in 150-200 words, illustrating their types, appearance and other information such as functionalities, usages, daily-life knowledge."
#     # gt["task_type"] = "Caption"
#     # SVQA
#     # gt['query'] = "Generate 5 single-round Q&As about 5 different object in rooms,considering diverse aspects like usage,material,belonged rooms and daily-life knowledge."
#     # gt["task_type"] = "VQA"
#
#     # Relation
#     gt['query'] = "Analyze the relationship of two object in the given scene point cloud. Generate a relation explanation."#For example,a dining table and a bowl are used for dining.Cupboard can be used to store plates.Remember relation explanation must be about two things and the object mentioned must be various and in the given scene point cloud."
#     gt["task_type"] = "Relation"
#
#     gt['id'] = i
#     gt['pcl'] = "scene/"+str(i)+".npy"
#     gt['src_dataset'] = "Mydata"
#     outjson.append(gt)
#





## Object GPT Test:SVQA,Relation,SCaption  ##
# import jsonlines
# GT = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/ClassificationCaption/Train_class/Classification.jsonl"))
# outjson = []
# for gt in GT:
#     #SVQA
#     # gt['query'] = "Generate 5 single-round Q&As about 5 different object in rooms,considering diverse aspects like usage,material,belonged rooms and daily-life knowledge."
#     # Relation
#     gt['query'] = "Analyze the relationship of two object in the given scene point cloud.Generate a relation explanation.For example,a dining table and a bowl are used for dining.Cupboard can be used to store plates.Remember relation explanation must be about two things and the object mentioned must be various and in the given scene point cloud."
#     # SCaption
#     gt['id'] = 'O'+gt['id'][1:]
#     # gt['query'] = "Write a detailed caption by classifying and describing different rooms in 150-200 words, illustrating their types, appearance and other information such as functionalities, usages, daily-life knowledge."
#     outjson.append(gt)



# # Object GPT Train:SVQA,Relation,SCaption  ##
# import jsonlines
# import re
# GT = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/ClassificationVQA/Train/Classification.jsonl"))
# # GT = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/ClassificationCaption/Train_class/Classification.jsonl"))
# outjson = []

# for gt in GT:
#     #SVQA
#     # gt['query'] = "Generate 5 single-round Q&As about 5 different object in rooms,considering diverse aspects like usage,material,belonged rooms and daily-life knowledge."
#     # SCaption
#     gt["text"] = gt["text"].replace('\ufffd', '')
#     if len(gt["text"])<=600 :
#         continue
#     name = re.sub(r".*\_", "", gt['id'])
#     name = re.sub(r"\d+.*", "", name)
#     gt['pcl'] = 'O'+gt['id'][1:]+".npy"
#     # gt["conversations"] = [{"from": "human",
#     #                         "value":f"Describe this object as detailed as possible, as if the object is right in front of you."},
#     #                        {
#     #                            "from": "gpt",
#     #                            "value": gt["text"]
#     #                        }
#     # ]
#     # gt['task_type'] = 'DescriptionObj3d'
#     gt["conversations"] = [{"from": "human",
#                             "value":f"You need to create three question-and-answer pairs centered around the object, ensuring that the context is interconnected. Format your response as a list,[Q1,A1,Q2,A2,Q3,A3]"},
#                            {
#                                "from": "gpt",
#                                "value": gt["text"]
#                            }
#     ]
#     gt['task_type'] = 'ConversationObj3d'
#     gt['src_dataset'] = "Mydata"
#     outjson.append(gt)

############################ 扩展的Agent Train ##

def Train_Agent():
    """
    Agent training data for 4 tasks (2-tool budget):
    - VisualGrounding_plus: DETECT -> (review select 1 idx) -> FINISH (output bbox)
    - Counting: DETECT -> (review select explain-indices) -> COUNT(tool, using ALL proposals) -> FINISH (ceil to gt_choices if present)
    - RoomDetection: DETECT -> (review outputs multiple room groups, each 2~8 idx) -> BBOX_UNION(tool, per-group) -> FINISH (output ALL rooms)
    - PositionRelation: DETECT -> (review select A/B) -> REL_DIR(tool) -> FINISH (choose final option / natural language)

    IMPORTANT:
    - Do NOT leak "Subtask=...".
    - RoomDetection asks: "Locate the locations of every room within the scene."
      and outputs multiple room instances (can have multiple of same type).
    - Counting: explain-indices are from top-20, but count uses ALL proposals; final uses gt_choices upward rounding.
    - ROOM_SPACE is 4 rooms: bedroom/kitchen/livingroom/bathroom.
    """

    import os, json, random, re
    from typing import Any, Dict, List, Optional, Tuple

    root = "/data/HTC/Data/dataset"
    result = root + "/Benchmark/temp.json"
    outjson = []

    # ---------- Load templates / GT ----------
    # Add VG_plus data loading for VisualGrounding_plus task
    try:
        vg_plus_path = "/data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/WholeTrain/VisualGrounding_plus.json"
        
        vg_plus_data = {}
        if os.path.exists(vg_plus_path):
            with open(vg_plus_path, "r") as f:
                v_data = json.load(f)
                for item in v_data:
                    # e.g., "scene/3.npy" or id="3", let's use id directly
                    sid = str(item.get("id", ""))
                    if not sid:
                        # fallback parse from pcl
                        pcl = item.get("pcl", "")
                        if "scene/" in pcl:
                            sid = pcl.split("/")[-1].replace(".npy", "")
                    if sid:
                        if sid not in vg_plus_data:
                            vg_plus_data[sid] = []
                        vg_plus_data[sid].append(item)
    except Exception as e:
        print(f"Warning: Could not load VisualGrounding_plus.json: {e}")
        vg_plus_data = {}

    Qc, Ac, GTc, _ = filepath('Counting')
    Q_Counting, Answer_Counting, GT_Counting = loading(Qc, Ac, GTc)  # list indexed by scene id (or dict-like)

    Qv, Av, GTv, _ = filepath('VisualGrounding')
    Q_VG, _, _ = loading(Qv, Av, Qv)

    Qp, Ap, GTp, _ = filepath('PositionRelation')
    Q_REL, _, _ = loading(Qp, Ap, Qp)

    Qr, Ar, GTr, _ = filepath('RoomDetection')
    Q_ROOM, _, GT_ROOM = loading(Qr, Ar, GTr)

    Qd, Ad, GTd, _ = filepath('Detection')
    Q_DET, _, _ = loading(Qd, Ad, Qd)  # Detection questions

    # Q_Classification is not standard in filepath (often uses direct path in test generation)
    q_class_path = "/data/HTC/Data/dataset/Benchmark/Task/Template/Q_Classification.json"
    with open(q_class_path, 'r') as qf:
        Q_CLASS = json.load(qf)

    # Detection proposals
    det_meta_path = "/data/HTC/Data/dataset/Benchmark/Task/GT/Detection.json"
    det_meta = {}
    with open(det_meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                det_meta.update(json.loads(line))
            except json.JSONDecodeError:
                continue

    # ---- 4-room space only (user confirmed) ----
    ROOM_SPACE = ["bedroom", "kitchen", "livingroom", "bathroom"]

    # Minimal room hint mapping (used only to synthesize supervision groups)
    ROOM_HINTS = {
        "bedroom":    ["bed", "door", "doorway", "wardrobe", "nightstand", "dresser"],
        "kitchen":    ["fridge", "stove", "sink", "cabinet", "countertop", "microwave"],
        "livingroom": ["sofa", "tv", "television", "door", "doorway", "coffeetable"],
        "bathroom":   ["toilet", "sink", "door", "doorway", "shower", "bathtub"],
    }

    # ---------- helpers ----------
    def norm(s: str) -> str:
        return ''.join(ch for ch in (s or '').lower() if ch.isalnum())

    def get_bbox(o: Dict[str, Any]) -> Optional[List[float]]:
        bb = o.get("BoundingBox", None)
        if isinstance(bb, list) and len(bb) == 6:
            return bb
        return None

    def center_of_bbox(bb: List[float]) -> Tuple[float, float, float]:
        # support both minmax or center formats (best-effort)
        # if minmax: [xmin,ymin,zmin,xmax,ymax,zmax]
        if bb[0] <= bb[3] and bb[1] <= bb[4] and bb[2] <= bb[5]:
            return ((bb[0] + bb[3]) / 2.0, (bb[1] + bb[4]) / 2.0, (bb[2] + bb[5]) / 2.0)
        # else treat as center format [cx,cy,cz,l,w,h]
        return (bb[0], bb[1], bb[2])

    def format_detect_list(det_topk: List[Dict[str, Any]]) -> str:
        lines = []
        for i, o in enumerate(det_topk):
            name = (o.get("name", "") or o.get("label", "") or "").lower()
            bbox = o.get("BoundingBox", None)
            lines.append(f"{name}{bbox}")
        return "\n".join(lines)

    def find_indices_by_keywords(det_list: List[Dict[str, Any]], keywords: List[str]) -> List[int]:
        kws = [norm(k) for k in (keywords or []) if k]
        if not kws:
            return []
        hits = []
        for i, o in enumerate(det_list):
            n = norm(o.get("name", "") or o.get("label", ""))
            for k in kws:
                if k and k in n:
                    hits.append(i)
                    break
        return hits

    def unique_name_indices(det_topk: List[Dict[str, Any]]) -> List[int]:
        cnt = {}
        for o in det_topk:
            n = norm(o.get("name", "") or o.get("label", ""))
            if not n:
                continue
            cnt[n] = cnt.get(n, 0) + 1
        return [i for i, o in enumerate(det_topk)
                if (n := norm(o.get("name", "") or o.get("label", ""))) and cnt.get(n, 0) == 1]

    def ceil_to_choices(v: int, choices: List[int]) -> int:
        if not choices:
            return v
        ch = sorted(int(x) for x in choices)
        for c in ch:
            if c >= v:
                return c
        return ch[-1]

    # ---- Position relation tool: getdir (your rule) ----
    def getdir(p0, p1):
        x0, y0, z0 = p0[:3]
        x1, y1, z1 = p1[:3]
        dis = 0.5
        if abs(y0 - y1) > dis and abs(x0 - x1) < dis and abs(z0 - z1) < dis:  # up/down
            if y0 > y1:
                return random.randint(240, 269), 0
            elif y0 < y1:
                return random.randint(210, 239), 0
        elif abs(y0 - y1) < 2 * dis and abs(x0 - x1) > dis and abs(z0 - z1) < dis:  # left/right
            if x0 > x1:
                return random.randint(60, 89), 0
            elif x0 < x1:
                return random.randint(30, 59), 0
        elif abs(y0 - y1) < 2 * dis and abs(x0 - x1) < dis and abs(z0 - z1) > dis:  # front/back
            if z0 > z1:
                return random.randint(90, 119), 0
            elif z0 < z1:
                return random.randint(120, 149), 0
        elif abs(y0 - y1) < 2 * dis and abs(x0 - x1) > dis and abs(z0 - z1) > dis:  # diagonal
            if z0 > z1 and x0 > x1:
                return random.randint(150, 179), 0  # a left-front of b
            elif z0 > z1 and x0 < x1:
                return random.randint(180, 209), 0  # a right-front of b
            elif z0 < z1 and x0 > x1:
                return random.randint(180, 209), 1  # b left-back of a
            elif z0 < z1 and x0 < x1:
                return random.randint(150, 179), 1  # b right-back of a
        else:
            return 0, -1

    # ---------- prompts (NO Subtask leakage) ----------
    def make_intent_prompt(user_question: str) -> str:
        return (
            "[AGENT_INTENT]\n"
            f"UserQuestion: {user_question}\n\n"
            # "You are an MLLM agent controller. Output ONE JSON only.\n"
            # "Schema:\n"
            # "{\n"
            # "  \"stage\":\"intent\",\n"
            # "  \"task\":\"VisualGrounding_plus|Counting|RoomDetection|PositionRelation\",\n"
            # "  \"focus\": {\"target\":\"\", \"A\":\"\", \"B\":\"\"}\n"
            # "}\n"
        )

    def make_review_prompt(user_question: str, det_topk: List[Dict[str, Any]]) -> str:
        detect_text = format_detect_list(det_topk)
        return (
            "[TOOL_RESULT]\n"
            "Tool=DETECT\n"
            f"Objects(topk={len(det_topk)}):\n{detect_text}\n\n"
            "[AGENT_REVIEW]\n"
            # f"UserQuestion: {user_question}\n\n"
            # "Output ONE JSON only.\n"
        )

    def make_finish_prompt(user_question: str, tool2_result_text: str) -> str:
        def simplify_bbox(bbox_text: str) -> str:
            import re
            def round_numbers(match):
                return f"{float(match.group()):.2f}"

            return re.sub(r"-?\d+\.\d+", round_numbers, bbox_text)

        simplified_result = simplify_bbox(tool2_result_text)
        return (
            "[TOOL_RESULT]\n"
            f"{simplified_result}\n\n"
            "[AGENT_FINISH]\n"
            # f"UserQuestion: {user_question}\n\n"
        )

    # ---------- tool simulators (for building supervised conversations) ----------
    def tool_count_all(all_objs: List[Dict[str, Any]], target: str) -> int:
        # use ALL proposals (not limited to 20)
        hits = find_indices_by_keywords(all_objs, [target])
        return len(hits)

    REL_TEMPLATE_PATH = "/data/HTC/Data/dataset/Benchmark/Task/Template/A_PositionRelation.json"

    @lru_cache(maxsize=1)
    def _rel_templates():
        with open(REL_TEMPLATE_PATH, "r") as f:
            d = json.load(f)
        return {str(k): str(v) for k, v in d.items()}

    def tool_rel_dir(det_topk: List[Dict[str, Any]], a_idx: int, b_idx: int) -> Dict[str, Any]:
        bbA = get_bbox(det_topk[a_idx]) or [0, 0, 0, 0, 0, 0]
        bbB = get_bbox(det_topk[b_idx]) or [0, 0, 0, 0, 0, 0]
        pA = center_of_bbox(bbA)
        pB = center_of_bbox(bbB)
        qid, flip = getdir(pA, pB)
        # 1) qid 直接索引模板
        tmpl = _rel_templates().get(str(qid), "")

        # 2) flip=1 就反一下（交换 C1/C2）
        if tmpl and int(flip) == 1:
            tmpl = tmpl.replace("C1", "__TMP__").replace("C2", "C1").replace("__TMP__", "C2")

        return {"qid": int(qid), "flip": int(flip), "text": tmpl}

    def tool_union_groups(det_topk: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # union bbox per group (minmax union)
        out = []
        for g in groups:
            indices = [int(i) for i in (g.get("indices") or [])]
            indices = [i for i in indices if 0 <= i < len(det_topk)]
            bbs = [get_bbox(det_topk[i]) for i in indices]
            bbs = [bb for bb in bbs if bb is not None]
            if not bbs:
                continue
            # union (minmax)
            mins = [float("inf")] * 3
            maxs = [float("-inf")] * 3
            for bb in bbs:
                # treat as minmax if plausible else convert from center
                if not (bb[0] <= bb[3] and bb[1] <= bb[4] and bb[2] <= bb[5]):
                    cx, cy, cz, l, w, h = bb
                    bb = [cx - l/2, cy - h/2, cz - w/2, cx + l/2, cy + h/2, cz + w/2]
                mins[0] = min(mins[0], bb[0]); mins[1] = min(mins[1], bb[1]); mins[2] = min(mins[2], bb[2])
                maxs[0] = max(maxs[0], bb[3]); maxs[1] = max(maxs[1], bb[4]); maxs[2] = max(maxs[2], bb[5])
            # 保留一位小数
            out.append({"room_label": g.get("room_label", "unknown"), "bbox": [round(mins[0], 1), round(mins[1], 1), round(mins[2], 1), round(maxs[0], 1), round(maxs[1], 1), round(maxs[2], 1)]})
        return out

    def split_room_into_instances(det_topk: List[Dict[str, Any]], base_indices: List[int], max_instances: int = 2) -> List[List[int]]:
        """
        Make multiple room instances for same label (best-effort) by simple spatial split.
        - If many indices, split by median X of bbox centers into up to 2 clusters.
        - Otherwise single instance.
        """
        idxs = [i for i in base_indices if 0 <= i < len(det_topk)]
        if len(idxs) < 4 or max_instances <= 1:
            return [idxs] if idxs else []
        xs = []
        for i in idxs:
            bb = get_bbox(det_topk[i])
            if bb is None:
                xs.append((i, 0.0))
            else:
                cx, _, _ = center_of_bbox(bb)
                xs.append((i, cx))
        xs.sort(key=lambda t: t[1])
        mid = len(xs) // 2
        g1 = [i for i, _ in xs[:mid]]
        g2 = [i for i, _ in xs[mid:]]
        groups = []
        if len(g1) >= 2:
            groups.append(g1)
        if len(g2) >= 2:
            groups.append(g2)
        return groups[:max_instances] if groups else [idxs]

    # ---------- Generate samples ----------
    sample_id = 0
    
    # [Task Control Switch]
    ENABLE_TASKS = {
        "VisualGrounding_plus", 
        "Counting", 
        "RoomDetection", 
        "PositionRelation",
        "Detection",
        "Classification"
    }

    # ENABLE_TASKS = ("PositionRelation")
    
    for sid in range(0, 100):   #460
        sid_str = str(sid)
        pcl_path = "scene/" + sid_str + ".npy"

        all_objs = det_meta.get(sid_str, [])
        if not all_objs:
            continue
        det_topk = all_objs[:20]
        if not det_topk:
            continue
        
        # -----------------------
        # 1) VisualGrounding_plus
        # -----------------------
        scene_vgps = vg_plus_data.get(sid_str, [])
        if "VisualGrounding_plus" in ENABLE_TASKS and scene_vgps:
            # Generate ALL instances for this scene rather than just one
            for vgp_obj in scene_vgps:
                vg_target = None
                vg_query = None
                selected_indices = []
                det_filtered = []
                vg_name_norm = ""
                
                convs = vgp_obj.get("conversations", [])
                if len(convs) >= 2:
                    human_val = convs[0].get("value", "")
                    gpt_val = convs[1].get("value", "")
                    
                    # parse obj index, e.g. "It is (obj3)."
                    m = re.search(r'obj(\d+)', gpt_val)
                    if m:
                        target_idx = int(m.group(1))
                        if 0 <= target_idx < len(all_objs):
                            vg_target = all_objs[target_idx]
                            vg_name = (vg_target.get("name", "") or vg_target.get("label", "") or "").lower()
                            vg_name_norm = norm(vg_name)
                            
                            vg_query = human_val
                            # Remove the specific prefix as requested
                            vg_query = vg_query.replace("In all objects, tell me which is ", "Which is ")
                            vg_query = vg_query.replace("In all objects, tell me which is", "Which is")
                            
                            # Find all objects of the same class in all_objs so the agent can choose
                            hits = find_indices_by_keywords(all_objs, [vg_name_norm])
                            hits = hits[:20]
                            if target_idx not in hits:
                                if len(hits) == 20:
                                    hits[-1] = target_idx
                                else:
                                    hits.append(target_idx)
                                hits = sorted(list(set(hits)))
                                
                            det_filtered = [all_objs[i] for i in hits]
                            try:
                                selected_indices = [hits.index(target_idx)]
                            except ValueError:
                                selected_indices = []

                if vg_query and vg_target and vg_name_norm:
                    qtemp = vg_query
                    user_q = vg_query # Already formatted
                    
                    intent_ans = {
                        "stage": "intent",
                        "task": "VisualGrounding_plus",
                        "focus": {"target": vg_name_norm},
                        "tool_plan": [
                            {"tool": "DETECT", "args": {}},
                            {"tool": "FINISH", "args": {}},
                        ],
                    }
                    review_ans = {
                        "stage": "review",
                        "summary": "Select the referred object indices, then finish with their bboxes.",
                        "selected_object_indices": selected_indices,
                        "room_groups": [],
                        "next_tool": {"tool": "FINISH", "args": {}},
                    }
                    
                    # Final text should include bboxes of ALL selected objects
                    bboxes = []
                    for idx in selected_indices:
                         b = get_bbox(det_filtered[idx])
                         if b:
                             bboxes.append(str(b))
                    final_text = " ".join(bboxes) if bboxes else "unknown"

                    conversations = [
                        {"from": "human", "value": make_intent_prompt(user_q)},
                        {"from": "gpt", "value": json.dumps(intent_ans, ensure_ascii=False)},
                        {"from": "human", "value": make_review_prompt(user_q, det_filtered)},
                        {"from": "gpt", "value": json.dumps(review_ans, ensure_ascii=False)},
                        {"from": "human", "value": make_finish_prompt(user_q, "Tool=FINISH\n(ready)")},
                        {"from": "gpt", "value": final_text},
                    ]
                    outjson.append(getsinglejson(sid_str, str(sample_id), pcl_path, conversations, "Agent3d"))
                    sample_id += 1

        # Fallback if no VG_plus data found for this scene (optional, removed to match your requirement of using all valid vg_plus_data)

        # -----------------------
        # 2) Counting
        # -----------------------
        gt_count = None
        if "Counting" in ENABLE_TASKS:
            try:
                gt_count = GT_Counting[sid]  # dict: {class: num} (if available)
            except Exception:
                gt_count = None

        if isinstance(gt_count, dict) and len(gt_count) > 0:
            keys = list(gt_count.keys())
            random.shuffle(keys)
            keys = keys[:1]
            for objclass in keys:
                obj_norm = norm(objclass)
                if not obj_norm:
                    continue
                qtemp = Q_Counting[str(random.randint(0, 29))]
                user_q = re.sub(r"{C}", objclass, qtemp) if "{C}" in qtemp else (qtemp + " " + objclass)

                # Use ALL objects to find matches (like VG_plus), ensuring targeted filtering
                hits = find_indices_by_keywords(all_objs, [obj_norm])
                
                # We need to make sure we don't exceed token limits but keep relevant objects
                # For counting, if we have > 20 matches, maybe just keep top 20
                hits = sorted(list(set(hits)))
                hits = hits[:20]
                det_filtered = [all_objs[i] for i in hits]
                
                # If no matches (shouldn't happen with correct logic), fallback 
                if not hits:
                     det_filtered = []

                # Update explain indices to be relative to det_filtered
                explain_new = list(range(len(det_filtered)))

                # tool COUNT uses ALL objs
                count_all = tool_count_all(all_objs, obj_norm)

                final_count = int(count_all)

                # Generate options similar to PositionRelation
                option_labels = ["(A)", "(B)", "(C)", "(D)"]
                
                # Logic from get_testCounting but adapted for 4 options and dynamic range
                # get_testCounting uses range(1, 10) for small counts.
                # Here we try to generate distractors around the final_count or within reasonable range.
                
                distractors = []
                pool = list(range(1, 10)) # Default small objects pool
                if final_count >= 10:
                    pool = list(range(max(1, final_count - 5), final_count + 6))
                
                if final_count in pool:
                    pool.remove(final_count)
                
                if len(pool) >= 3:
                     distractors = random.sample(pool, 3)
                else:
                     # Fallback if pool is too small (e.g. count is 1, pool is 2..9? no, verify logic)
                     # If count=1, pool=2..9. If count=100, pool=95..106 remove 100.
                     while len(distractors) < 3:
                        d = random.randint(1, 10)
                        if d != final_count and d not in distractors:
                            distractors.append(d)
                
                options = [str(x) for x in distractors]
                # Insert correct answer at random position
                answer_pos = random.randint(0, 3)
                options.insert(answer_pos, str(final_count))
                
                # Append options to user_q
                user_q += "\nOptions:"
                for i, opt in enumerate(options):
                    user_q += f"\n{option_labels[i]} {opt}"

                intent_ans = {
                    "stage": "intent",
                    "task": "Counting",
                    "focus": {"target": obj_norm},
                    "tool_plan": [
                        {"tool": "DETECT", "args": {}},
                        {"tool": "COUNT", "args": {"target": obj_norm}},
                        {"tool": "FINISH", "args": {}},
                    ],
                }
                review_ans = {
                    "stage": "review",
                    "summary": "Select indices for explanation (top-20), then call COUNT over all proposals.",
                    "selected_object_indices": explain_new,
                    "room_groups": [],
                    "next_tool": {"tool": "COUNT", "args": {"target": obj_norm}},
                }
                tool2_text = f"Tool=COUNT\nTarget={obj_norm}\ncount_all={count_all}\nfinal_count={final_count}"

                # Generate natural language final answer like Train_Counting
                ans_temp = Answer_Counting[str(random.randint(0, 29))]
                ans_text = re.sub(r"{C}", objclass, ans_temp)
                ans_text = re.sub(r"{N}", str(final_count), ans_text)

                conversations = [
                    {"from": "human", "value": make_intent_prompt(user_q)},
                    {"from": "gpt", "value": json.dumps(intent_ans, ensure_ascii=False)},
                    {"from": "human", "value": make_review_prompt(user_q, det_filtered)},
                    {"from": "gpt", "value": json.dumps(review_ans, ensure_ascii=False)},
                    {"from": "human", "value": make_finish_prompt(user_q, tool2_text)},
                    {"from": "gpt", "value": ans_text},
                ]
                outjson.append(getsinglejson(sid_str, str(sample_id), pcl_path, conversations, "Agent3d"))
                sample_id += 1

        # -----------------------
        # 3) RoomDetection (ALL rooms, possibly multiple instances per type)
        # -----------------------
        # Always generate (doesn't rely on GT_ROOM), because the training signal is index grouping + union.
        if "RoomDetection" in ENABLE_TASKS:
            user_q = "Locate the locations of every room within the scene."
        
            # Load Ground Truth rooms for the current scene (using sid_str)
            # Load Ground Truth rooms for the current scene (using sid_str)
            # Assuming GT_ROOM is a dict {sid: {room_name: bbox, ...}}
            # Or a list where index corresponds to scene. But earlier code uses `loading`.
            # Let's inspect `Train_RoomDetection` or `getanswerRoomDe` logic.
            # In Train_RoomDetection: Q, A, GT, result = filepath('RoomDetection'); Question, Answer, GT = loading(Q, A, GT)
            # GT is from `loading`.
            
            # Let's try to get GT rooms for this scene
            gt_rooms_scene = {}
            # GT_ROOM loaded above using `loading` might be Dict[str, Dict] where key is scene_id
            if sid_str in GT_ROOM:
                 gt_rooms_scene = GT_ROOM[sid_str]
            elif int(sid_str) in GT_ROOM:
                 gt_rooms_scene = GT_ROOM[int(sid_str)]

            # RoomDetection: Input 20 random objects
            # We already have `det_topk` which is top-20 from `all_objs`.
            # If we want "random" 20, we can sample from `all_objs` if len > 20.
            # But `det_topk = all_objs[:20]` is currently used.
            # User request: "RoomDetection输入二十个随机物体".
            # Let's shuffle `all_objs` and take 20.
            det_room_input = all_objs[:]
            random.shuffle(det_room_input)
            det_filtered = det_room_input[:20]
            
            # Since we changed component list, we must re-calculate `review_ans` (selected indices)
            # The indices for room detection are groups of objects inside rooms.
            # We need to re-run the "Find objects inside this GT bbox" logic using `det_filtered`.
            
            intent_ans = {
                "stage": "intent",
                "task": "RoomDetection",
                "focus": {"target": ""},
                "tool_plan": [
                    {"tool": "DETECT", "args": {}},
                    {"tool": "BBOX_UNION", "args": {}},
                    {"tool": "FINISH", "args": {}},
                ],
            }
    
            room_groups = []
            
            final_room_bboxes = [] # List of {"room_label":..., "bbox":...}
            
            # Iterate over GT rooms for this scene
            # gt_rooms_scene is likely { "bedroom": [x,y,z,l,w,h], "bathroom": ... }
            if gt_rooms_scene:
                for r_name, r_bbox in gt_rooms_scene.items():
                    # Clean room name (remove possible instance suffixes if any, assuming standard names)
                    # ... (same logic)
                    valid_label = "unknown"
                    for space in ROOM_SPACE:
                        if space in r_name.lower():
                            valid_label = space
                            break
                    if valid_label == "unknown":
                        continue # Skip unknown rooms
                    
                    final_bbox = r_bbox
                    if isinstance(r_bbox, list) and len(r_bbox) > 0 and isinstance(r_bbox[0], dict):
                         try:
                             final_bbox = point2box(r_bbox)
                         except Exception as e:
                             continue
                    
                    if not isinstance(final_bbox, list) or len(final_bbox) != 6 or isinstance(final_bbox[0], dict):
                         continue
                    
                    final_room_bboxes.append({"room_label": valid_label, "bbox": final_bbox})
                    
                    # For Review step: Find objects inside this GT bbox
                    # USING det_filtered now
                    idxs_in_room = []
                    for idx, obj in enumerate(det_filtered):
                         obj_bb = get_bbox(obj)
                         if not obj_bb: continue
                         cx, cy, cz = obj_bb[0], obj_bb[1], obj_bb[2]
                         if isinstance(final_bbox, dict):
                             continue
                         
                         try:
                            rcx, rcy, rcz, rl, rw, rh = [float(x) for x in final_bbox]
                            if (abs(cx - rcx) <= rl/2) and (abs(cy - rcy) <= rh/2) and (abs(cz - rcz) <= rw/2):
                                idxs_in_room.append(idx)
                         except Exception as e:
                            continue
                    
                    if not idxs_in_room:
                         # Fallback: pick objects matching keywords IN DET_FILTERED
                         fk = [norm(x) for x in (ROOM_HINTS.get(valid_label, []) or []) if x]
                         idxs_in_room = find_indices_by_keywords(det_filtered, fk)
                    
                    if not idxs_in_room:
                         idxs_in_room = [0] # Last resort
                    
                    idxs_in_room = idxs_in_room[:8]
                    room_groups.append({"room_label": valid_label, "indices": idxs_in_room})
            else:
                 room_groups = [{"room_label": "livingroom", "indices": list(range(min(4, len(det_filtered))))}]
                 final_room_bboxes = []
    
            review_ans = {
                "stage": "review",
                "summary": "Group objects for each room instance and compute unions to localize every room.",
                "selected_object_indices": [],
            }
    
            # tool2 union per group -> produce ALL room bboxes
            # union_out = tool_union_groups(det_topk, room_groups) # Not used for GT generation actually, we use final_room_bboxes
            union_out = final_room_bboxes
    
            # ... (text gen logic same) ...
            
            # Build final answer (one per sentence). Keep compact and parse-friendly.
            final_lines = []
            for item in union_out:
                lbl = item.get("room_label", "unknown")
                bbox = item.get("bbox", None)
                
                # Ensure bbox is a list of numbers
                if bbox is not None and isinstance(bbox, list) and len(bbox) == 6:
                    try:
                        # Filter out any non-numeric items or dicts
                        safe_bbox = []
                        for x in bbox:
                            if isinstance(x, (int, float)):
                                safe_bbox.append(float(x))
                            elif isinstance(x, str):
                                safe_bbox.append(float(x))
                            else:
                                raise ValueError("Not a number")
                        
                        bb = [round(x, 2) for x in safe_bbox]
                    except (ValueError, TypeError):
                        bb = None
                else:
                    bb = None
    
                if bb is None:
                    continue
                final_lines.append(f"{lbl} {bb}")
            final_text = "\n".join(final_lines) if final_lines else "unknown"
    
            # tool2_text construction needs to be safe as well
            tool2_parts = ["Tool=BBOX_UNION"]
            for x in union_out:
                lbl = x.get('room_label')
                bbox = x.get('bbox')
                if not isinstance(bbox, list): continue
                try:
                     b_rounded = [round(float(z), 2) for z in bbox]
                     tool2_parts.append(f"{lbl}{b_rounded}")
                except:
                     continue
            
            tool2_text = "\n".join(tool2_parts)
    
            conversations = [
                {"from": "human", "value": make_intent_prompt(user_q)},
                {"from": "gpt", "value": json.dumps(intent_ans, ensure_ascii=False)},
                {"from": "human", "value": make_review_prompt(user_q, det_filtered)},
                {"from": "gpt", "value": json.dumps(review_ans, ensure_ascii=False)},
                {"from": "human", "value": make_finish_prompt(user_q,'')},
                {"from": "gpt", "value": final_text},
            ]
            outjson.append(getsinglejson(sid_str, str(sample_id), pcl_path, conversations, "Agent3d"))
            sample_id += 1

        # -----------------------
        # 4) PositionRelation (DETECT -> REL_DIR -> FINISH)
        # -----------------------
        uniq_idxs = unique_name_indices(det_topk)
        if "PositionRelation" in ENABLE_TASKS and len(uniq_idxs) >= 2:
            a_idx, b_idx = random.sample(uniq_idxs, 2)

            a_name = (det_topk[a_idx].get("name", "") or det_topk[a_idx].get("label", "") or "").lower()
            b_name = (det_topk[b_idx].get("name", "") or det_topk[b_idx].get("label", "") or "").lower()
            a_norm, b_norm = norm(a_name), norm(b_name)

            if a_norm and b_norm:
                # 1) 生成问题：用“模板库里某个问题qid”生成 query（qid 来自几何工具）
                #    但如果 bbox 缺失 / getdir 返回无效，就退化为随机关系模板(30~269)中的一个
                bbA = get_bbox(det_topk[a_idx])
                bbB = get_bbox(det_topk[b_idx])

                if bbA is None or bbB is None:
                    qid = random.randint(30, 269)
                    flip = 0
                else:
                    rel_out = tool_rel_dir(det_topk, a_idx, b_idx)  # 只调用一次
                    qid = int(rel_out.get("qid", 0))
                    flip = int(rel_out.get("flip", 0))
                    if qid <= 0:
                        qid = random.randint(30, 269)
                        flip = 0

                qtemp = Q_REL.get(
                    str(qid),
                    Q_REL.get(str(random.randint(30, 269)), "Describe the relation between {C1} and {C2}.")
                )

                # flip=1：问题里 C1/C2 交换（等价“反正”）
                if flip == 1:
                    user_q_base = qtemp.replace("{C1}", b_name).replace("{C2}", a_name)
                    # Prepare answer template swap
                    ans_tmpl = _rel_templates().get(str(qid), "C1 is related to C2.")
                    ans_tmpl = ans_tmpl.replace("C1", "__TMP__").replace("C2", "C1").replace("__TMP__", "C2")
                    correct_ans_text = ans_tmpl.replace("C1", a_name).replace("C2", b_name)
                else:
                    user_q_base = qtemp.replace("{C1}", a_name).replace("{C2}", b_name)
                    ans_tmpl = _rel_templates().get(str(qid), "C1 is related to C2.")
                    correct_ans_text = ans_tmpl.replace("C1", a_name).replace("C2", b_name)

                # Generate 3 distractors
                # Strategy: pick 3 other random relation IDs, check they are not same as `qid`
                # Only use valid keys from _rel_templates which are typically 1-28 or 30-269
                all_rel_keys = list(_rel_templates().keys())
                distractors = []
                while len(distractors) < 3:
                    rq = random.choice(all_rel_keys)
                    if rq == str(qid): continue # skip correct
                    # construct text
                    d_tmpl = _rel_templates()[rq]
                    # random flip for distractor
                    if random.random() > 0.5:
                        d_text = d_tmpl.replace("C1", b_name).replace("C2", a_name) # Logic maybe wrong but it's a distractor
                    else:
                        d_text = d_tmpl.replace("C1", a_name).replace("C2", b_name)
                    if d_text not in distractors and d_text != correct_ans_text:
                        distractors.append(d_text)
                
                # Assemble Options
                options = distractors + [correct_ans_text]
                random.shuffle(options)
                correct_idx = options.index(correct_ans_text)
                option_labels = ["(A)", "(B)", "(C)", "(D)"]
                
                # Construct final user query with options
                user_q = user_q_base + "\nOptions:"
                for i, opt in enumerate(options):
                    user_q += f"\n{option_labels[i]} {opt}"

                # 2) intent：不泄露 Subtask 字段，只让模型输出结构化工具计划
                
                # Filter DET to only include A and B (and maybe some distractors if we wanted, but user said "only targets")
                # User request: "PositionRelation输入两个目标"
                # Find all objects matching intent keywords (a_norm, b_norm)
                # Ensure specifically the chosen pair (a_idx, b_idx) is included if they happen to be missed (unlikely)
                hits = find_indices_by_keywords(all_objs, [a_norm, b_norm])
                
                # a_idx and b_idx are from det_topk which is all_objs[:20]. So they are valid indices in all_objs.
                if a_idx not in hits: hits.append(a_idx)
                if b_idx not in hits: hits.append(b_idx)
                
                hits = sorted(list(set(hits)))
                hits = hits[:20] 
                det_filtered = [all_objs[i] for i in hits]
                
                # Identify new indices for the specific pair in the filtered list
                try:
                    new_a_idx = hits.index(a_idx)
                    new_b_idx = hits.index(b_idx)
                except ValueError:
                    # Fallback (should not happen due to append above)
                    new_a_idx = 0
                    new_b_idx = 1 if len(det_filtered) > 1 else 0

                intent_ans = {
                    "stage": "intent",
                    "task": "PositionRelation",
                    "focus": {"target": [a_norm, b_norm]},
                    "tool_plan": [
                        {"tool": "DETECT", "args": {}},
                        {"tool": "REL_DIR", "args": {}},
                        {"tool": "FINISH", "args": {}},
                    ],
                }

                # 3) review：监督模型选 A/B 两个 index，并触发 REL_DIR
                # Use new indices
                review_ans = {
                    "stage": "review",
                    "summary": "Pick indices for the queried pair, then call REL_DIR.",
                    "selected_object_indices": [new_a_idx, new_b_idx],
                    "room_groups": [],
                    "next_tool": {"tool": "REL_DIR", "args": {"A_idx": new_a_idx, "B_idx": new_b_idx}},
                }

                # 4) tool2 输出（REL_DIR）：把 qid/flip 给 FINISH
                # user request: tool2_text output convert to complete sentence
                tool2_text = correct_ans_text

                # 5) FINISH 的监督答案：输出完整句子（或者也可以带上选项 A/B/C/D，视训练目标而定）
                
                final_rel_sentence = correct_ans_text

                conversations = [
                    {"from": "human", "value": make_intent_prompt(user_q)},
                    {"from": "gpt", "value": json.dumps(intent_ans, ensure_ascii=False)},
                    {"from": "human", "value": make_review_prompt(user_q, det_filtered)},
                    {"from": "gpt", "value": json.dumps(review_ans, ensure_ascii=False)},
                    {"from": "human", "value": make_finish_prompt(user_q, tool2_text)},
                    {"from": "gpt", "value": final_rel_sentence},
                ]
                outjson.append(getsinglejson(sid_str, str(sample_id), pcl_path, conversations, "Agent3d"))
                sample_id += 1

        # -----------------------
        # 5) Detection
        # -----------------------
        if "Detection" in ENABLE_TASKS:
            # Random question template
            que_num = random.randint(0, 29)
            user_q_det = Q_DET.get(str(que_num), "Can you tell me what these items are?") + " With obj0:name0, obj1:name1... form of answers. "
            
            # Simple DETECT -> FINISH plan
            intent_ans_det = {
                "stage": "intent",
                "task": "Detection",
                "focus": {"target": ""},
                "tool_plan": [
                    {"tool": "DETECT", "args": {}},
                    {"tool": "FINISH", "args": {}},
                ],
            }
            
            # All identified objects
            review_ans_det = {
                "stage": "review",
                "summary": "Output all detected objects in the requested format.",
                "selected_object_indices": list(range(len(det_topk))),
                "room_groups": [],
                "next_tool": {"tool": "FINISH", "args": {}},
            }
            
            # Construct final answer format
            final_text_parts = []
            for i, o in enumerate(det_topk):
                name = (o.get("name", "") or o.get("label", "") or "").lower()
                final_text_parts.append(f"(obj{i}):{name}!")
            final_text_det = " ".join(final_text_parts) if final_text_parts else "unknown"
            
            conversations_det = [
                {"from": "human", "value": make_intent_prompt(user_q_det)},
                {"from": "gpt", "value": json.dumps(intent_ans_det, ensure_ascii=False)},
                {"from": "human", "value": make_review_prompt(user_q_det, det_topk)},
                {"from": "gpt", "value": json.dumps(review_ans_det, ensure_ascii=False)},
                {"from": "human", "value": make_finish_prompt(user_q_det, "Tool=FINISH\n(ready)")},
                {"from": "gpt", "value": final_text_det},
            ]
            outjson.append(getsinglejson(sid_str, str(sample_id), pcl_path, conversations_det, "Agent3d"))
            sample_id += 1

        print(sid, "Agent OK")

    # -----------------------
    # 6) Classification
    # -----------------------
    if "Classification" in ENABLE_TASKS:
        try:
            class_train_data = json.load(open("/data/HTC/Data/dataset/object_add/my_train.json"))
            ALL_names_list = json.load(open("/data/HTC/Data/dataset/object_add/my_names.json", 'r'))
            
            for path in class_train_data[:100]:
                src_id = re.sub(r"_.*","",path)
                classname = re.sub(r".*\d_", "", path)
                classname = re.sub(r"\d.*", "", classname)
                pcl_path = "Objects/"+path+".npy"
                
                que_num = random.randint(0, 29)
                base_q = Q_CLASS.get(str(que_num), "What's the 3D point cloud about?")
                
                random_number = random.sample(range(0, len(ALL_names_list)), 5)
                random_class = [ALL_names_list[i] for i in random_number]
                while classname in random_class:
                    random_number = random.sample(range(0, len(ALL_names_list)), 5)
                    random_class = [ALL_names_list[i] for i in random_number]
                    
                answer_pos = random.randint(0, 5)
                random_class.insert(answer_pos, classname)
                gt_choices = random_class
                
                answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) ","4":" (E) ","5":" (F) "}
                query = base_q + " \n Options: "
                for i_idx, c_name in enumerate(gt_choices):
                    query += answer_query[str(i_idx)] + c_name
                    
                final_answer = answer_query[str(answer_pos)].strip() + " " + classname
                
                intent_ans_cls = {
                    "stage": "intent",
                    "task": "Classification",
                    "focus": {"target": ""},
                    "tool_plan": [
                        {"tool": "FINISH", "args": {}},
                    ],
                }
                
                conversations_cls = [
                    {"from": "human", "value": make_intent_prompt(query)},
                    {"from": "gpt", "value": json.dumps(intent_ans_cls, ensure_ascii=False)},
                    {"from": "human", "value": make_finish_prompt(query, "Tool=FINISH\n(ready)")},
                    {"from": "gpt", "value": final_answer},
                ]
                outjson.append(getsinglejson(src_id, str(sample_id), pcl_path, conversations_cls, "Agent3d"))
                sample_id += 1
            print("Classification Agent OK")
        except Exception as e:
            print(f"Classification Agent error: {e}")

    return result, outjson


############################ 扩展的VG Test ##
def VG_Test():
    Description = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/VQA/VQA.jsonl"))
    VGobj = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/VisualGrounding.json"))
    outjson = []
    Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
    new_classdict = {}

    for i in Detection:
        key = list(i.keys())[0]
        if int(list(i.keys())[0]) < 460 or int(list(i.keys())[0])>=500:
            continue
        # new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]
        new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Detection_class]
        new_classdict[key] = new_classlist
    for i in Description:
        gt = {}
        vg = VGobj[i['pcl'][:3]]
        ovnamelist= [name for name,other in vg.items()]
        gt["pcl"] = "scene/"+i["pcl"][:-4]+".npy"
        gt["id"] = i["pcl"][:-4]
        question = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_VisualGrounding.json"))
        gt["query"] = "Which is "+ i["text"][0].lower()+i["text"][1:]
        scene = new_classdict[i["pcl"][:-4]]
        gt["obj_num"] = None
        for index,box in enumerate(scene):
            if box["BoundingBox"] == i["box"]:
                gt["obj_num"] = index
                gt["bbox"] = i["box"]
        if gt["obj_num"]==None:
            continue
        if  gt["obj_num"]>20:
            continue
        outjson.append(gt)

    return outjson




# 扩展的VG Train ##
def VG_Train():
    Description = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/VG_description/Detection_description.jsonl"))
    outjson = []

    Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
    new_classdict = {}
    for i in Detection:
        key = list(i.keys())[0]
        if int(list(i.keys())[0]) >= 460:
            continue
        new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]
        new_classdict[key] = new_classlist
    for i in Description:
        gt = {}
        gt["pcl"] = "scene/"+i["pcl"][:-4]+".npy"
        gt["id"] = i["pcl"][:-4]
        question = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_VisualGrounding.json"))
        query = "Which is a  "+ i["text"][0].lower()+i["text"][1:]
        scene = new_classdict[i["pcl"][:-4]]
        for index,box in enumerate(scene):
            if box["BoundingBox"] == i["box"]:
                obj_num = index
        if obj_num>20:
            continue
        gt["conversations"]= [
            {
                "from": "human",
                "value": query
            },
            {
                "from": "gpt",
                "value": "It is (obj"+str(obj_num) +")."
            }
        ]
        gt["task_type"] = "VisualGrounding3d"
        outjson.append(gt)
    return outjson





"""
训练Instruction tuning data
"""
#Classification
# result, outjson = Train_Classification()
#Counting
# result, outjson = Train_Counting()
# Detection
# result, outjson = Train_Detection()
#VisualGrounding
# result, outjson = Train_VisualGrounding()
#RoomDetection
# result, outjson = Train_RoomDetection()
#Navigation
# result, outjson = Train_Navigation()
#PositionRelation
# result, outjson = Train_PositionRelation()


#Counting Agent Parser
result, outjson = Train_Agent()

"""
测试Instruction tuning data
"""
# outjson = VG_Train()
# outjson = VG_Test()
# #Classification
# result, outjson = Test_Classification()
# #Counting
# result, outjson = Test_Counting()
#Detection
# result, outjson = Test_Detection()
#Multi_Classification
# result, outjson = Multi_Classification()
#VisualGrounding
# result, outjson = Test_VisualGrounding()
#RoomDetection
# result, outjson = Test_RoomDetection()
#Navigation
# result, outjson = Test_Navigation()
#PositionRelation
# result, outjson = Test_PositionRelation()
result = "/data/HTC/Data/dataset/Benchmark/Agent_v1_demo.json"
with open(result, 'w') as f:
    # 把列表写入到文件里，转换成json格式
    json.dump(outjson, f, indent=4)
#


#
# exsiting = []
#
# scene_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/pro_Detection.json"
# sce = open(scene_path,"r")
# for item in jsonlines.Reader(sce):
#     exsiting.append(list(item.keys())[0])
#
# #多线程控制
#
# max_processes = 10
# pool = multiprocessing.Pool(processes=max_processes)
#
# num_jobs = 30000  # 总共要执行的任务数
#
# for i in range(0, num_jobs):
#     if str(i) in exsiting:
#         continue
#     #     # process_item(i)
#     # 启动一个新进程来执行 worker_function
#     pool.apply_async(process_item, args=(i,))
#
# # 关闭进程池，不再接受新任务
# pool.close()
#
# # 等待所有进程完成
# pool.join()
# print("所有进程已完成")
