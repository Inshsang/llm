#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json

import numpy as np
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence
import pickle
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines

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

class LAMMDataset(Dataset):
    """LAMM Dataset"""

    # def __init__(self, data_file_path: str, vision_root_path: str, choose: bool, vision_type="pcl"):
    #     """Initialize supervised datasets
    #
    #     :param str data_file_path: path of conversation file path
    #     :param str vision_root_path: vision root path
    #     :param str vision_type: type of vision data, defaults to 'image', image / pcl
    #     """
    #     super(LAMMDataset, self).__init__()
    #     self.vision_type = vision_type
    #     self.choose = choose
    #
    #     with open(data_file_path, "r") as fr:
    #         json_data = json.load(fr)
    #
    #     self.vision_path_list, self.caption_list, self.task_type_list = [], [], []
    #     Multi_class_scene2class= {str(i):[0,0,0,0,0,0] for i in range(500,10000)}
    #     for item in json_data:
    #         if not vision_type in item:
    #             continue
    #         one_vision_name, one_caption = item[vision_type], item["conversations"]
    #         task_type = item["task_type"] if "task_type" in item else "normal"
    #
    #         if not one_vision_name.startswith("/"):
    #             one_vision_path = os.path.join(vision_root_path, one_vision_name)
    #         else:
    #             one_vision_path = one_vision_name
    #
    #         if item['task_type'] == 'Detection3d':
    #             Multi_class_scene2class[item['src_id']] = item['box']
    #         self.vision_path_list.append(one_vision_path)
    #         self.caption_list.append(one_caption)
    #         self.task_type_list.append(task_type)
    #
    #     self.num2name = json.load(open("/media/kou/Data3/htc/dataset/Object/my_names.json"))
    #     with open("/media/kou/Data3/htc/dataset/Object/my_train_8192pts_fps.dat", 'rb') as f:
    #         self.list_of_objpoints = pickle.load(f)
    #
    #
    #     # with open("/media/kou/Data3/htc/dataset/cut_scene_train.dat", 'rb') as f:
    #     #     self.detection_gt = pickle.load(f)
    #     # self.detection_gt = torch.tensor(self.detection_gt)
    #
    #     with open("/media/kou/Data3/htc/dataset/cut_scene_500_label.dat", 'rb') as f:
    #         self.detection_gt_500_num = pickle.load(f)
    #     with open("/media/kou/Data3/htc/dataset/cut_scene_train_label.dat", 'rb') as f:
    #         self.detection_gt_num = pickle.load(f)
    #
    #     self.detection_gt_num = torch.tensor(self.detection_gt_num)
    #     self.detection_gt_500_num = torch.tensor(self.detection_gt_500_num)
    #     num = torch.sum(self.detection_gt_num)
    #     num_500 = torch.sum(self.detection_gt_500_num)
    #     self.detection_gt = ["/media/kou/Data3/htc/Detection_obj/"+str(i)+".npy" for i in range(num)]
    #     self.detection_gt_500 = ["/media/kou/Data3/htc/Detection_obj/" + str(i) + "_500.npy" for i in range(num_500)]
    #
    #     self.list_of_objpoints =[["/media/kou/Data3/htc/Objects_8192_npy/points/"+str(i)+".npy" for i in range(9500)],
    #                              ["/media/kou/Data3/htc/Objects_8192_npy/labels/"+str(i)+".npy" for i in range(9500)]]
    #
    #     # self.map_class2points={self.vision_path_list[i]: "/media/kou/Data3/htc/Objects_8192_npy/points/"+str(i)+".npy"
    #     #                   for i in range(len(self.vision_path_list))}
    #     # self.map_class2labels = {self.vision_path_list[i]: "/media/kou/Data3/htc/Objects_8192_npy/labels/" + str(i) + ".npy"
    #     #                     for i in range(len(self.vision_path_list))}
    #
    #     class_map = json.load(open("/media/kou/Data3/htc/dataset/Object/my_train.json"))
    #     self.map_class2points = {vision_root_path+'/Objects/'+class_map[i]+".npy": "/media/kou/Data3/htc/Objects_8192_npy/points/" + str(i) + ".npy" for i in range(len(class_map))}
    #     self.map_class2labels = {vision_root_path+'/Objects/'+class_map[i]+".npy": "/media/kou/Data3/htc/Objects_8192_npy/labels/" + str(i) + ".npy" for i in range(len(class_map))}
    #     self.map_scene2points=["/media/kou/Data3/htc/Objects_8192_npy/points/"+str(i)+".npy" for i in range(len(self.vision_path_list))]
    #
    #     p0 = 0
    #     self.pos = []
    #     for p in self.detection_gt_num:
    #         self.pos.append(int(p0))
    #         p0 += p
    #     p0 = 0
    #     self.pos_500 = []
    #     for p in self.detection_gt_500_num:
    #         self.pos_500.append(int(p0))
    #         p0 += p
    #
    #
    #     self.scene_gt = {}
    #     index = -1
    #     index_500 = -1
    #     with open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json", "r") as G:
    #         jsonlines_data = jsonlines.Reader(G)
    #         for lines in jsonlines_data:
    #             id = next(iter(lines))
    #
    #             if int(id)<10000 and int(id) >= 500:
    #                 #增加单个场景物体
    #                 inclass_box = []
    #                 inclass_class = []
    #                 index += 1
    #                 bbox = lines[id]
    #                 for i in bbox:
    #                     if not len(i):
    #                         continue
    #                     if i['name'].lower() in Class_ALL:
    #                         inclass_box.append(i['name'].lower())
    #                         inclass_class.append(i['BoundingBox'])
    #
    #                 all_num = int(self.detection_gt_num[index])
    #                 start_num = self.pos[index]
    #                 inclass_points=self.detection_gt[start_num:start_num+all_num]
    #                 self.scene_gt[id] = {'classes':inclass_box,'boxes':inclass_class,'points':inclass_points,"Multi_class":Multi_class_scene2class[id]}
    #
    #             if int(id) < 500:
    #                 #增加单个场景物体
    #                 inclass_box = []
    #                 inclass_class = []
    #                 index_500 += 1
    #                 bbox = lines[id]
    #                 for i in bbox:
    #                     if not len(i):
    #                         continue
    #                     if i['name'].lower() in Class_ALL:
    #                         inclass_box.append(i['name'].lower())
    #                         inclass_class.append(i['BoundingBox'])
    #                 all_num = int(self.detection_gt_500_num[index_500])
    #                 start_num = self.pos_500[index_500]
    #                 inclass_points=self.detection_gt_500[start_num:start_num+all_num]
    #                 self.scene_gt[id] = {'classes':inclass_box,'boxes':inclass_class,'points':inclass_points}
    #
    #     # self.choosen_num = len(self.detection_gt_num)
    #     # if self.choose:
    #     #     choose_tensor = self.detection_gt_num <= 12
    #     #     numall = int(choose_tensor.sum())
    #     #     self.choosen = list(np.array(choose_tensor))
    #     #     self.choosen_num = len(self.choosen)
    #     #     new_pos = numall*[None]
    #     #     p0, index = 0,0
    #     #     for f,p in zip(self.choosen,self.detection_gt_num):
    #     #         if f:
    #     #             new_pos[index] = p0
    #     #             index += 1
    #     #         p0 += int(p)
    #     #     self.pos = new_pos
    #     #     self.detection_gt_label = self.detection_gt_num[choose_tensor]
    #         # self.vision_path_list = [path for f, path in zip(self.choosen,self.vision_path_list) if f]
    #         # self.caption_list = [caption for f, caption in zip(self.choosen, self.caption_list) if f]
    #         # self.task_type_list = [task_type for f, task_type in zip(self.choosen, self.task_type_list) if f]
    #         # self.list_of_objpoints[0] = [points for f, points in zip(self.choosen, self.list_of_objpoints[0]) if f]
    #         # self.list_of_objpoints[1] = [points_label for f, points_label in zip(self.choosen, self.list_of_objpoints[1]) if f]
    #         # self.scene_gt = [c for i,c in zip(self.choosen,self.scene_gt) if i]
    #     print(f"[!] collect {len(self.vision_path_list)} samples for training")
    def __init__(self, data_file_path: str, vision_root_path: str, choose: bool, vision_type="pcl"):
        """Initialize supervised datasets

        :param str data_file_path: path of conversation file path
        :param str vision_root_path: vision root path
        :param str vision_type: type of vision data, defaults to 'image', image / pcl
        """
        super(LAMMDataset, self).__init__()
        self.vision_type = vision_type
        self.choose = choose

        with open(data_file_path, "r") as fr:
            json_data = json.load(fr)

        self.vision_path_list, self.caption_list, self.task_type_list = [], [], []
        Multi_class_scene2class= {str(i):[0,0,0,0,0,0] for i in range(0,10000)}
        for item in json_data:
            if not vision_type in item:
                continue
            one_vision_name, one_caption = item[vision_type], item["conversations"]
            task_type = item["task_type"] if "task_type" in item else "normal"

            if not one_vision_name.startswith("/"):
                one_vision_path = os.path.join(vision_root_path, one_vision_name)
            else:
                one_vision_path = one_vision_name

            if item['task_type'] == 'Detection3d':
                Multi_class_scene2class[item['src_id']] = item['box']
            self.vision_path_list.append(one_vision_path)
            self.caption_list.append(one_caption)
            self.task_type_list.append(task_type)

        self.num2name = json.load(open("/data/HTC/Data/dataset/object_add/my_names.json"))
        with open("/data/HTC/Project/Point-BERT/data/ModelNet/modelnet40_normal_resampled/my_train_1024pts_fps.dat", 'rb') as f:
            self.list_of_objpoints = pickle.load(f)

        class_map = json.load(open("/data/HTC/Project/Point-BERT/data/ModelNet/modelnet40_normal_resampled/my_train.json"))
        # self.map_class2points = {vision_root_path+'/object_npy/'+class_map[i]+".npy": "/media/kou/Data3/htc/Objects_8192_npy/points/" + str(i) + ".npy" for i in range(len(class_map))}
        # self.map_class2labels = {vision_root_path+'/object_npy/'+class_map[i]+".npy": "/media/kou/Data3/htc/Objects_8192_npy/labels/" + str(i) + ".npy" for i in range(len(class_map))}
        self.map_class2points = {}
        self.map_class2labels = {}
        class_target_root = '/data/HTC/Data/dataset/object_1024_npy'
        for i in range(len(class_map)):
            sample_name = class_map[i]
            sample_file = sample_name if sample_name.endswith('.npy') else sample_name + '.npy'
            target_file = os.path.join(class_target_root, sample_file)

            # 兼容历史与当前两种输入路径格式
            key_variants = [
                os.path.join(vision_root_path, 'Objects', sample_file),
                os.path.join(vision_root_path, 'object_1024_npy', sample_file),
                sample_file,
                sample_name,
            ]
            for key in key_variants:
                self.map_class2points[key] = target_file
                self.map_class2labels[key] = target_file
        self.scene_gt = {}
        index = -1
        self.VG = json.load(open("/data/HTC/Data/dataset/Benchmark/Task/GT/VisualGrounding.json", "r"))
        with open("/data/HTC/Data/dataset/Benchmark/Task/GT/Detection.json", "r") as G:
            jsonlines_data = jsonlines.Reader(G)
            for lines in jsonlines_data:
                id = next(iter(lines))
                if int(id)<10000:
                    inclass_box = []
                    inclass_class = []
                    index += 1
                    bbox = lines[id]
                    for i in bbox:
                        if not len(i):
                            continue
                        if i['name'].lower() in Class_ALL:
                            inclass_box.append(i['name'].lower())
                            inclass_class.append(i['BoundingBox'])
                    self.scene_gt[id] = {'classes':inclass_box,'boxes':inclass_class,"Multi_class":Multi_class_scene2class[id]}


        print(f"[!] collect {len(self.vision_path_list)} samples for training")

    def __len__(self):
        """get dataset length

        :return int: length of dataset
        """
        return len(self.vision_path_list)


    def __getitem__(self, i):
        """get one sample"""
        task_type = self.task_type_list[i]
        vision_path = self.vision_path_list[i]

        def _scene_id_from_path(path: str) -> str:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in self.scene_gt:
                return stem
            if '_' in stem:
                prefix = stem.split('_', 1)[0]
                if prefix in self.scene_gt:
                    return prefix
            digits = ''.join(ch for ch in stem if ch.isdigit())
            if digits in self.scene_gt:
                return digits
            return stem

        if self.task_type_list[i] in ['Classification3d','DescriptionObj3d','ConversationObj3d']:#Detection,Counting,'Classification3d',PositionRelation,VG,RoomDetection,Navigation
            key = vision_path
            if key not in self.map_class2points:
                key = os.path.basename(key)
            points_path = self.map_class2points[key]
            label_path = self.map_class2labels[key]
        elif task_type in ['Detection3d']:
            scene_id = _scene_id_from_path(vision_path)
            points_path = self.scene_gt[scene_id]['Multi_class']
            label_path = self.scene_gt[scene_id]['classes']
            vision_path = '/data/HTC/Data/dataset/Benchmark/data/scene/' + scene_id + '.ply'
        elif task_type in ['Agent3d']:
            scene_id = _scene_id_from_path(vision_path)
            points_path = self.scene_gt[scene_id]['boxes']
            label_path = self.scene_gt[scene_id]['classes']
            vision_path = '/data/HTC/Data/dataset/Benchmark/data/scene/' + scene_id + '.ply'
        else:
            scene_id = _scene_id_from_path(vision_path)
            points_path = self.scene_gt[scene_id]['boxes']
            label_path = self.scene_gt[scene_id]['classes']
            vision_path = '/data/HTC/Data/dataset/Benchmark/data/scene/' + scene_id + '.ply'
        return dict(
            vision_paths=vision_path,
            output_texts=self.caption_list[i],
            vision_type=self.vision_type,
            task_type=task_type,
            points_path = points_path,
            label_path=label_path
        )

    def collate(self, instances):
        """collate function for dataloader"""
        vision_paths, output_texts, task_type ,points_path,label_path= tuple(
            [instance[key] for instance in instances]
            for key in ("vision_paths", "output_texts", "task_type","points_path","label_path")
        )
        return dict(
            vision_paths=vision_paths,
            output_texts=output_texts,
            vision_type=self.vision_type,
            task_type=task_type,
            points_path = points_path,
            label_path=label_path
        )
