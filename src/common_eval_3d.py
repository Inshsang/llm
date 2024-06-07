import argparse
import os
import json
import numpy as np
from utils import *
from tqdm import tqdm
from Loading import LAMM_EVAL_3D
import random
from torch.utils.data import DataLoader, Dataset
def Navigation(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    panish = 1    #每个节点的损失约束
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['positions']
        text = pred['text']
        points = parse_bbox_3d_Nav(text)
        cnt += 1
        len_a = len(gt_objects)
        len_b = len(points)
        if len_a < len_b:
            short = gt_objects
            long = points
        else:
            short = points
            long = gt_objects
        difference = cal_path_3d(short,long)
        if difference < panish*len(gt_objects):
            score += 1

    print(score / cnt)

def grounding3d_eval(dataset, pred_data, thres=0.25):
    score = 0
    cnt = 0
    scene_num = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt["object"]
        text = pred['text']
        points = parse_bbox_3d_Vis(text)
        if len(gt_objects) > 10:
            continue
        # if len(points) > 10:
        #     continue
        cnt += len(gt_objects)  # gt_objects,points
        for object_info in gt_objects:
            if (not classification_acc(object_info['name'].lower(), text.lower())) and (not (object_info['name'].lower() in text.lower())):
                continue
            # if not (object_info['label'] in text):
            #     continue
            for index, point in enumerate(points):
                # iou = cal_iou_3d(object_info['bbox'], point)
                iou = cal_aro_3d(object_info['BoundingBox'], point)
                if iou > thres:
                    score += 1
                    break
        scene_num += 1
    print(scene_num,score / cnt)


#直接对Detection专家检测
# def grounding3d_eval(dataset, pred_data, thres=0.5):
#     score = 0
#     cnt = 0
#     metadata = json.load(open("/media/kou/Data1/htc/LAMM/data/metadata/Detection.json"))
#     for gt, pred in tqdm(zip(dataset, metadata), ncols=40):
#         gt_objects = gt["object"]
#         text = metadata[gt["id"]]
#         if len(gt_objects) > 30:
#             continue
#         text_name = [i["name"] for i in text]
#         # text = parse_bbox_3d_Vis(text)
#         cnt += len(gt_objects)#gt_objects,points
#         for object_info in gt_objects:
#             if not (object_info['name'].lower() in text_name):
#                 continue
#             # if not (object_info['label'] in text):
#             #     continue
#             for index, point in enumerate(text):
#                 #iou = cal_iou_3d(object_info['bbox'], point)
#                 iou = cal_aro_3d(object_info['BoundingBox'], point['BoundingBox'])
#                 if iou > thres:
#                     score += 1
#                     break
#     print(score / cnt)

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

def Rgrounding3d_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['object']
        text = pred['text']
        bboxes = parse_bbox_3d_Vis(text)
        cnt += len(gt_objects)#gt_objects,bboxes
        # for object_info in gt_objects:
        # if not classification_acc(gt_objects['label'], text):
        #     continue
        # for bbox in bboxes:

        for object_info in gt_objects:
            #判断房间分类
            if (not classification_acc(object_info['label'], text)) and (not (object_info['label'].lower() in text.lower())):
                continue
            for index, point in enumerate(bboxes):
                iou = cal_iou_3d(object_info['bbox'], point)

                if iou > 0.5:
                    score += 1
                    break
    print(score / cnt)

def Vgrounding3d_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['object']
        text = pred['text']
        bboxes = parse_bbox_3d_Vis(text)
        cnt += 1#gt_objects,bbox
        # for object_info in gt_objects:
        # if not classification_acc(gt_objects['label'], text):
        #     continue
        # for bbox in bboxes:
        if len(bboxes) != 1:
            continue
        if len(bboxes[0]) != 6:
            continue
        iou = cal_aro_3d(gt_objects, bboxes[0])
        # if iou > 0:
        #     print(iou)
        if iou > thres:
            score += 1
    print(score / cnt)

def grounding3d(dataset, pred_data):
    Vgrounding3d_eval(dataset, pred_data, thres=0.25)
    # Vgrounding3d_eval(dataset, pred_data, thres=0.5)

CHOICE = ['A', 'B', 'C', 'D', 'E', 'F']         # 6 choices in total


def VQAvisionacc(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-F]\)?\W|the answer is \(?[A-F]\)?\W')
    pattern_2 = re.compile(r'option [A-F]')
    pattern_3 = re.compile(r'\([A-F]\)')
    def check_text(text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    testnum = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        pred_text = pred['text']
        pred_text = pred_text
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        if len(res_1) != 0:
            if check_option(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) != 0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
        score += tmp_score
        testnum += 1
    print('vision: {}'.format(score / testnum))

def Positoinacc(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-D]\)?\W|the answer is \(?[A-D]\)?\W')
    pattern_2 = re.compile(r'\([A-D]\)')
    pattern_3 = re.compile(r'[A-D]')


    def check_text(text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-2]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    testnum = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        pred_text = pred['text']
        pred_text = pred_text
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)

        if len(res_1) != 0:
            if check_pattern2(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) != 0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
        else :
            Addition = {'0':"A",'1':"B",'2':"C",'3':"D"}
            choice = random.randint(0,3)
            choice = Addition[str(choice)]
            if check_option(choice, gt_char):
                tmp_score = 1.0

        score += tmp_score
        testnum += 1
    print('vision: {}'.format(score / testnum))
    
def Counting(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-F]\)?\W|the answer is \(?[A-F]\)?\W')
    pattern_2 = re.compile(r'ANSWER: [A-F]')
    pattern_3 = re.compile(r'\([A-F]\)')
    TwoEnglish = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                  '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
                  '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen',
                  '19': 'nineteen', '20': 'twenty'}
    def check_text(text, choices, gt_id):
        text = text.lower()
        if str(choices[gt_id]) not in text and TwoEnglish[str(choices[gt_id])] not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if str(choice) in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        pred_text = pred['text']
        pred_text = pred_text
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        if len(res_1) != 0:
            if check_option(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) != 0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
        score += tmp_score
    print('vision: {}'.format(score / len(dataset)))

dataset2evalfunc = {
    'Detection': grounding3d_eval,
    'MyData': VQAvisionacc,
    'ScanRefer': grounding3d,
    'ScanQA_multiplechoice': VQAvisionacc,
    'Counting': Counting,
    'Class': VQAvisionacc,
    'PositionRelation':Positoinacc,
    'VG':grounding3d,
    'Navigation':Navigation,
    'RoomDetection':Rgrounding3d_eval
}

def collate_fn(batch):
    res = dict()
    keys = batch[0].keys()
    for key in keys:
        res[key] = [data[key] for data in batch]
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Mydata")#Lamm,Mydata
    parser.add_argument("--task-name", default="Detection")#Detection,Counting,Class,PositionRelation,VG,RoomDetection,Navigation
    parser.add_argument('--answer-file', default=r"/media/kou/Data1/htc/LAMM/answers")
    parser.add_argument('--base-data-path', default=r"/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Test")
    args = parser.parse_args()
   
    dataset_name = args.dataset_name
    task_name = args.task_name
    dataset = LAMM_EVAL_3D(args.base_data_path,
                           dataset_name,
                           task_name
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False,
                            collate_fn=collate_fn)
    dataset = dataloader.dataset

    eval_func = dataset2evalfunc[task_name]


    # if args.answer_file.endswith('.jsonl'):
    # if task_name == 'Navigation':
    #     jonal = r'G:\event\htc\LAMM\answers\Navigation\Navigation_Mydata.jsonl'
    # if task_name == 'VG':
    #     jonal = r'G:\event\htc\LAMM\answers\VG\VG_Mydata.jsonl'
    # if task_name == 'Counting':
    #     jonal = r'G:\event\htc\LAMM\answers\Counting\Counting.jsonl'
    # if task_name == 'Class':
    #     jonal = r'G:\event\htc\LAMM\answers\Class\Class.jsonl'
    if task_name == 'Detection':
        jonal = r'/media/kou/Data1/htc/LAMM/answers/Detection.jsonl'
        import jsonlines
        pred_data = []

        with open(jonal, 'rb') as f:
            for item in jsonlines.Reader(f):
                pred_data.append(item)
    elif args.answer_file.endswith('.json'):
        pred_data = json.load(open(args.answer_file,'rb'))
    else:
        file_ext = '.json'
        file_name = task_name + '_' + dataset_name + file_ext
        args.answer_file = os.path.join(args.answer_file,task_name, file_name)
        pred_data = json.load(open(args.answer_file, 'rb'))
    print(f'Eval [{args.answer_file}] on {dataset_name}')
    eval_func(dataset, pred_data)
