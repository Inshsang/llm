import json
import os
from torch.utils.data import Dataset
import numpy as np
# from system_msg import common_task2sysmsg, locating_task2sysmsg


common_dataset2task = {
    'ScanNet_Lamm': 'Detection',
    'ScanRefer': 'VG',
    'ScanQA': 'SVQA',
    'ScanQA_multiplechoice': 'SVQA',
    'Counting': 'Counting',
    'Class': 'Class',
    'RoomDetection':'RoomDetection',
}


class LAMM_EVAL_3D(Dataset):
    def __init__(self,
                 base_data_path,
                 dataset_name,
                 task_name):
        self.base_data_path = base_data_path
        self.dataset_name = dataset_name
        self.task_name = task_name
        # self.task_name = common_dataset2task[self.dataset_name]
        print(self.dataset_name,self.task_name)
        self.system_msg =self.task_name+ '3D'
        if self.task_name=='Detection' and self.dataset_name=="Lamm":
            json_path = "/media/kou/Data1/htc/LAMM_v0/data/3D_Benchmark/meta_file/Detection_ScanNet.json"
        elif self.task_name=='Detection' and self.dataset_name !="Lamm":
            json_path ="/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Test/Detection.json"
        elif self.task_name == 'VisualGrounding' and self.dataset_name != "Lamm":
            json_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Test/VisualGrounding.json"
        else:
            json_path = "//media/kou/Data1/htc/LAMM_v0/data/3D_Benchmark/meta_file/VG_ScanRefer.json"
        self.data = json.load(open(json_path, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        if 'pcl' in data_item:
            data_item['pcl'] = os.path.join(self.base_data_path, data_item['pcl'])
        return data_item

    def __repr__(self) -> str:
        repr_str = '{}_{}\n\nSYSTEM_MSG:{}'
        return repr_str.format(self.task_name, self.dataset_name, self.system_msg)


