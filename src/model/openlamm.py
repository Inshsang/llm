import io
import json
import os
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn.utils import rnn
import open3d as o3d
import conversations
from header import *
from transformers import StoppingCriteria, StoppingCriteriaList
import logging
logger = logging.getLogger(__name__)
import numpy as np
from .CLIP import load as load_clip
from .EPCL import build_epcl_encoder
from .pointbert.point_encoder import PointTransformer
from .modeling_lightllm import LlamaLightForCausalLM
from .modeling_llama import LlamaForCausalLM
from .utils.pcl_utils import MEAN_COLOR_RGB, RandomCuboid, random_sampling
from .utils.data import transform_vision_data
from .utils.utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


VISION_TAGS = {
    "pos": {"image": "<image>", "pcl": "<pcl>"},
    "sov": {"image": "<Img>", "pcl": "<Pcl>"},
    "eov": {"image": "</Img>", "pcl": "</Pcl>"},
}

def farthest_point_sample(point, npoint=8192):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def cut_point_cloud(point_cloud, bbox_list):
    cut_parts = []
    for bbox in bbox_list:
        # Extracting xyzwlh from bbox
        x, y, z, width, length, height = bbox

        # Finding points within the bbox
        mask = np.stack((
            point_cloud[:, 0] >= x - width / 2,
            point_cloud[:, 0] <= x + width / 2,
            point_cloud[:, 1] >= y - length / 2,
            point_cloud[:, 1] <= y + length / 2,
            point_cloud[:, 2] >= z - height / 2,
            point_cloud[:, 2] <= z + height / 2),axis=0
        )
        mask = np.all(mask, axis=0)

        # Applying the mask to extract points
        cut_part = point_cloud[mask]

        while (len(cut_part)<8192):
            cut_part = interpolate_points(cut_part)
        cut_part = cut_part[np.random.choice(cut_part.shape[0], 8192, replace=False)]
        cut_parts.append(cut_part)

    return cut_parts

class LAMMStoppingCriteria(StoppingCriteria):
    def __init__(self, stops, input_ids):
        """intialize stopping criteria

        :param list stops: list of stop tokens
        :param list input_ids: input ids
        """
        super().__init__()
        self.stops = [torch.tensor(stop).to('cuda') for stop in stops]
        self.stop_flag = [0] * input_ids.shape[0]

    def check_stop(self, input_ids):
        """check whether to stop generation

        :param list input_ids: input token ids
        :return bool: stop or not
        """
        for stop in self.stops:
            if torch.all((stop == input_ids[-len(stop):])).item():
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """call function of stop creteria

        :param torch.LongTensor output_ids: output token ids
        :return bool: stop or not
        """
        flag = 1
        for id, output_id in enumerate(output_ids):
            if self.stop_flag[id] == 1:
                continue
            if self.check_stop(output_id):
                self.stop_flag[id] = 1
            else:
                flag = 0
        if flag == 1:
            return True
        return False


def build_one_instance(tokenizer, conversation, vision_type="image"):
    """build one instance for training; text part

    :param class tokenizer: text tokenizer
    :param list conversation: list of conversation
    :param str vision_type: type of vision data, defaults to 'image'
    :raises Exception: Exception if wrong role included
    :return list: conversation text list, input token ids, target token ids
    """
    pos = VISION_TAGS["pos"][vision_type]
    eov = VISION_TAGS["eov"][vision_type]
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn["from"]
        if i == 0:  # the first human turn
            assert role == "human"
            turn["value"] = (
                turn["value"].replace(f"{pos}\n", "").replace(f"\n{pos}", "")
            )
            text = f"{eov} " + turn["value"] + "\n### Assistant:"
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(
                one_input_id
            )  # do not perform loss regression on human prompt
        else:
            if role == "human":
                text = "Human: " + turn["value"] + "\n### Assistant:"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == "gpt":
                text = turn["value"] + "\n###"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception("Wrong Role!!!")
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids
    """'</Pcl> Find and give coordinates for objects within an indoor point cloud.
    ### Assistant:Situated at the [13.77, 0.32, 0.99, 2.81, 0.63, 1.0] coordinates within the point cloud, there exists an object, classified as DiningTable.
    ###'
    """
def process_batch_instance(
    tokenizer, batch_of_conversations, max_tgt_len, vision_type="image"
):
    """build one batch of instance for training
    :param class tokenizer: text tokenizer
    :param list batch_of_conversations: batch of conversations
    :param int max_tgt_len: max token length of after vision tokens
    :param str vision_type: type of vision data, defaults to 'image'
    :return list: input token ids, target token ids, attention mask
    """
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(
            tokenizer, conversation, vision_type=vision_type
        )#单论对话结尾本不应该是###
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    target_ids = rnn.pad_sequence(
        batch_target_ids, batch_first=True, padding_value=-100
    )
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def make_prompt_start(use_system=False, vision_type="image", task_type="normal"):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str vision_type: type of visio data, defaults to 'image'
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    PROMPT_START = f'### Human: {VISION_TAGS["sov"][vision_type]}'
    if use_system:
        if task_type == "normal":
            return f"{conversations.default_conversation.system}\n\n" + PROMPT_START
        else:
            return [
                f"{conversations.conversation_dict[task]}\n\n" + PROMPT_START
                for task in task_type
            ]
    else:
        return PROMPT_START

def interpolate_points(point_cloud, target_points=8192):
    n_points = len(point_cloud)

    while n_points < target_points:
        # 计算每个点与其最近邻点的距离
        distances = np.sum((point_cloud[:, np.newaxis] - point_cloud) ** 2, axis=-1)
        np.fill_diagonal(distances, np.inf)  # 避免将点与自身匹配

        # 找到每个点的最近邻点索引
        nearest_indices = np.argmin(distances, axis=1)

        # 对每个点进行插值
        interpolated_points = []
        for i, idx in enumerate(nearest_indices):
            interpolated_points.append((point_cloud[i] + point_cloud[idx]) / 2)

        # 添加插值点到原始点云中
        point_cloud = np.concatenate([point_cloud, np.array(interpolated_points)])

        # 更新点的数量
        n_points = len(point_cloud)

    # 如果点的数量超过目标值，随机选择一些点
    # if n_points > target_points:
    #     indices = np.random.choice(n_points, target_points, replace=False)
    #     point_cloud = point_cloud[indices]

    return point_cloud

class LAMMPEFTModel(nn.Module):
    """LoRA for LAMM model"""

    def __init__(self, **args):
        super(LAMMPEFTModel, self).__init__()
        self.args = args
        self.max_obj_len = args["max_obj_len"]
        self.vision_type = args["vision_type"] if "vision_type" in args else "pcl"
        self.train_stage = args["train_stage"]
        encoder_pretrain = (
            args["encoder_pretrain"] if "encoder_pretrain" in args else "clip"
        )
        assert encoder_pretrain in [
            "clip",
            "epcl",
        ], f"Encoder_pretrain: {encoder_pretrain} Not Implemented"
        encoder_ckpt_path = (
            args["encoder_ckpt_path"]
            if not encoder_pretrain == "clip"
            else "~/.cache/clip/ViT-L-14.pt"
        )
        vicuna_ckpt_path = args["vicuna_ckpt_path"]

        use_system = args["use_system"] if "use_system" in args else False
        # stage = args["stage"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print(
        #     f"Initializing [{encoder_pretrain}] visual encoder from {encoder_ckpt_path} [{device}]..."
        # )

        # -1 for last embedding; -2 for transformer output
        self.vision_feature_type = args["vision_feature_type"]
        self.num_vision_token = args["num_vision_token"]
        self.encoder_pretrain = encoder_pretrain

        # TODO: Make sure the number of vision tokens is correct
        if self.encoder_pretrain.lower() == "clip":
            clip_encoder, self.visual_preprocess = load_clip("ViT-L/14", device=device)
            self.visual_encoder = clip_encoder.visual
            if self.vision_feature_type == "global":  # global feature from CLIP
                self.vision_hidden_size = 768
                self.num_vision_token = 1
                assert self.num_vision_token == 1, "Only 1 global token is available!"
            elif self.vision_feature_type == "local":  # patch features from CLIP ViT
                self.vision_hidden_size = 384
                self.num_vision_token = min(
                    self.num_vision_token, 256
                )  # may cut partial tokens

        elif self.encoder_pretrain.lower() == "epcl":
            # PCL data Processing
            self.use_color = (
                self.args["use_color"] if "use_color" in self.args else False
            )
            self.use_height = (
                self.args["use_height"] if "use_height" in self.args else False
            )
            self.num_points = (
                self.args["num_points"] if "num_points" in self.args else 200000
            )

            # Pointbert
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", "PointTransformer_base_8192point.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if self.use_color:
                point_bert_config.model.point_dims = 6

            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=False)
            # logger.info(f"Using {self.point_backbone.point_dims} dim of points.")
            # self.point_backbone.load_checkpoint("/media/kou/Data1/htc/Point-BERT/ckpts/Point-BERT.pth")

            # load state dict
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args['local_rank']}
            state_dict = torch.load("/media/kou/Data1/htc/Point-BERT/experiments/PointTransformer_8192point/ModelNet_models/default/ckpt-last.pth", map_location=map_location)
            # parameter resume of base model
            # if args.local_rank == 0:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
            self.point_backbone.load_state_dict(base_ckpt, strict=True)

            self.point_backbone.to(device)

            # freeze vision encoder
            for name, param in self.point_backbone.named_parameters():
                param.requires_grad = False
            self.point_backbone.eval()

            if self.vision_feature_type == "global":
                raise NotImplementedError("Global feature not implemented for EPCL")
            else:
                self.vision_hidden_size = 1024
                self.num_vision_token = self.num_vision_token
            # self.visual_encoder = build_epcl_encoder(
            #     pretrain=True, store_path=encoder_ckpt_path, device=device
            # )

        else:
            raise NotImplementedError(
                f"Encoder {self.encoder_pretrain} not implemented!"
            )

        # # freeze vision encoder
        # for name, param in self.visual_encoder.named_parameters():
        #     param.requires_grad = False
        # self.visual_encoder.eval()
        print("Visual encoder initialized.")

        print(f"Initializing language decoder from {vicuna_ckpt_path} ...")
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
        )

        if args.get('answers_dir', False):
            self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            pass
            # self.llama_model = LlamaLightForCausalLM(
            #     batch_size=self.args['bs'],
            #     max_input_len=1024,
            #     max_output_len=args['max_tgt_len'],
            #     weight_dir=vicuna_ckpt_path,
            #     lora_path=args['delta_ckpt_path'],
            #     lora_config=peft_config,
            # )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
            self.llama_model = get_peft_model(self.llama_model, peft_config)

        # self.llama_proj = nn.Linear(256, self.llama_model.config.hidden_size)
        self.llama_proj = nn.Sequential(
            # nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.llama_model.config.hidden_size)
        )

        if self.train_stage == 1:
            #冻结llm
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()

            self.llama_proj = nn.Sequential(
                # nn.Linear(self.trans_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.llama_model.config.hidden_size)
            )
            print("########################Initial llama_proj##########################")

        elif self.train_stage == 2:
            # 加载保存的参数
            llama_proj = torch.load("/media/kou/Data1/htc/LAMM/ckpt/llama_projcetion/llama_proj_v1.pth")
            processed_llama = {key.replace("llama_proj.", ""): value for key, value in llama_proj.items()}
            # 加载参数到llama_pro层
            self.llama_proj.load_state_dict(processed_llama)
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj.eval()
            print("Froeze llama_proj.")
        elif self.train_stage == 3:
            #冻结llm
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()

            self.llama_proj = nn.Sequential(
                # nn.Linear(self.trans_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.llama_model.config.hidden_size)
            )
            # self.llama_proj = nn.Linear(
            #     256, self.llama_model.config.hidden_size
            # )
            print("########################Initial llama_proj##########################")

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            vicuna_ckpt_path, use_fast=False
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print("Language decoder initialized.")

        # self.llama_proj = nn.Linear(
        #     self.vision_hidden_size, self.llama_model.config.hidden_size
        # )

        print("LLaMa projection layer initialized.")

        self.max_tgt_len = args["max_tgt_len"]
        self.use_system = use_system
        self.use_flash_attn = args.get('use_flash_attn', False)
        self.use_xformers = args.get('use_xformers', False)
        self.device = torch.cuda.current_device()

    def encode_image(self, image_paths):
        """encode images to llama inputs

        :param tupe image_paths: (bsz, )
        :return tensor, tensor: input feature to llama, attention mask to llama
        """
        if self.encoder_pretrain == "clip":
            inputs = self.load_and_transform_image_data_clip(
                image_paths, self.device
            )  # bsz x 3 x 224 x 224
            inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32
            inputs_llama = self.clip_encode_image(inputs)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
                self.device
            )  # bsz x 1/256
            return inputs_llama, atts_llama
        else:
            raise NotImplementedError("Encoder not implemented!")

    def encode_image_object(self, images):
        """encoder loaded image objects"""
        if self.encoder_pretrain == "clip":
            inputs = transform_vision_data(
                images, self.device
            )  # bsz x 3 x 224 x 224
            inputs_llama = self.clip_encode_image(inputs)  # bsz x 1/256 x llama_size
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
                self.device
            )  # bsz x 1/256
            return inputs_llama, atts_llama
        else:
            raise NotImplementedError(
                "Encoder pretrain [{}] not implemented".format(self.encoder_pretrain)
            )

    def test_encode_pcl(self, pcl_paths, obj_list,list_of_objpoints):
        # load pcl data
        inputs = self.load_and_transform_pcl_data(
            pcl_paths, self.device
        )  # bsz x 40000 x 3
        inputs = inputs[0]
        embeddings = []
        for vision in obj_list:
            x,y,z,width,length,height = np.array(vision["BoundingBox"])
            # Finding points within the bbox
            mask = np.stack((
                inputs[:, 0] >= x - width / 2,
                inputs[:, 0] <= x + width / 2,
                inputs[:, 1] >= y - length / 2,
                inputs[:, 1] <= y + length / 2,
                inputs[:, 2] >= z - height / 2,
                inputs[:, 2] <= z + height / 2), axis=0
            )
            mask = np.all(mask, axis=0)
            # Applying the mask to extract points
            cut_part = inputs[mask]
            # # 可视化点云
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(cut_part)
            # o3d.visualization.draw_geometries([pcd])
            if 'Objects_npy' in pcl_paths[0]:
                cut_part = list_of_objpoints
            if len(cut_part)==0:
                # print("One empyt object")
                cut_part = inputs
            while (len(cut_part) < 8192):
                newcut_part = cut_part+0.00001
                cut_part = np.concatenate((cut_part, newcut_part), axis=0)
                # cut_part = interpolate_points(cut_part)
            cut_part = cut_part[np.random.choice(cut_part.shape[0], 8192, replace=False)]

            with torch.no_grad():
                if self.vision_feature_type == "global":
                    raise NotImplementedError("Global feature not implemented for pcl")
                elif self.vision_feature_type == "local":
                    obj = torch.tensor(cut_part).to(self.llama_model.dtype).to(self.device)
                    embedding = self.point_backbone(obj.unsqueeze(dim=0))
                    embedding = self.llama_proj(embedding)
                    embeddings.append(embedding)

        atts_llama = torch.ones(embedding.size()[:-1], dtype=torch.long).to(
            self.device
        )  # bsz x 1/256
        return embeddings, atts_llama

    def encode_pcl(self, pcl_paths):
        # load pcl data
        inputs = self.load_and_transform_pcl_data(
            pcl_paths, self.device
        )  # bsz x 40000 x 3

        inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32
        with torch.no_grad():
            if self.vision_feature_type == "global":
                raise NotImplementedError("Global feature not implemented for pcl")
            elif self.vision_feature_type == "local":
                embeddings = self.point_backbone(inputs)
                # embeddings = self.visual_encoder(inputs)[1][
                #     :, : self.num_vision_token
                # ]  # bsz x 256 x 1024;
                # image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(
                #     self.llama_model.dtype
                # )  # bsz*num vision token x 1024
        inputs_llama = self.llama_proj(embeddings).reshape(
            -1, 1, self.llama_model.config.hidden_size
        )  # bsz x num_vision_token x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            self.device
        )  # bsz x 1/256
        return inputs_llama, atts_llama

    def encode_obj_pcl(self, device,obj_list):
        with torch.no_grad():
            if self.vision_feature_type == "global":
                raise NotImplementedError("Global feature not implemented for pcl")
            elif self.vision_feature_type == "local":
                objs = torch.tensor(np.asarray(obj_list)).to(self.llama_model.dtype).to(device)
                embedding = self.point_backbone(objs)

        atts_llama = torch.ones(embedding.size()[:-1], dtype=torch.long).to(
            self.device
        )  # bsz x 1/256
        return embedding, atts_llama

    def clip_encode_image(self, inputs):
        inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32

        if self.vision_feature_type == "global":
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)  # bsz x 768
            image_embeds = embeddings.to(self.llama_model.dtype)
            inputs_llama = self.llama_proj(image_embeds).unsqueeze(
                1
            )  # bsz x 1 x llama_size
        elif self.vision_feature_type == "local":
            with torch.no_grad():
                embeddings = self.visual_encoder.forward_patch_features(inputs)[
                    :, : self.num_vision_token
                ]  # bsz x self.num_vision_token x 1024
            image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(
                self.llama_model.dtype
            )  # bsz*num vision token x 1024
            inputs_llama = self.llama_proj(image_embeds).reshape(
                -1, self.num_vision_token, self.llama_model.config.hidden_size
            )  # bsz x num_vision_token x llama_size
        else:
            raise NotImplementedError(
                "{} not Implemented".format(self.vision_feature_type)
            )
        return inputs_llama

    def load_and_transform_image_data_clip(self, image_paths, device):
        if image_paths is None:
            return None
        image_ouputs = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                image = Image.open(image_path)
            elif image_path.startswith("http://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                print("can not load image: ", image_path)
            image_output = self.visual_preprocess(image).to(device)  # 3 x 224 x 224
            image_ouputs.append(image_output)
        return torch.stack(image_ouputs, dim=0)  # B x 3 x 224 x 224

    def load_and_transform_pcl_data(self, pcl_paths, device):
        if pcl_paths is None:
            return None
        pcl_output = []
        for pcl_path in pcl_paths:
            if 'scene' in pcl_path:
                pcl_path = pcl_path[:-4] + '.ply'
                mesh_vertices = o3d.io.read_point_cloud(pcl_path)
                point_cloud = np.asarray(mesh_vertices.points)
                pcl_output.append(point_cloud)
            else:
                pcl_path = pcl_path[:-4] + '.npy'
                point_cloud = np.load(pcl_path)[:,:3]
                pcl_output.append(point_cloud)
            # pcl_output.append(torch.from_numpy(point_cloud))
        # return torch.stack(pcl_output, dim=0).to(device)  # bsz x num_points x 3
        return pcl_output

    def prompt_wrap(
        self, img_embeds, input_ids, target_ids, attention_mask, use_system, task_type
    ):
        """
        input_ids, target_ids, attention_mask: bsz x s2
        """
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2
        self.num_vision_token = img_embeds.size()[1]
        batch_size = len(img_embeds)

        # return list of headers if multiple tasks
        p_before = make_prompt_start(
            use_system=use_system, vision_type=self.vision_type, task_type=task_type
        )
        if isinstance(p_before, list):
            p_before_tokens = [
                self.llama_tokenizer(p, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .to(self.device)
                for p in p_before
            ]
            # TODO: test in batch
            p_before_token_ids = rnn.pad_sequence(
                p_before_tokens,
                batch_first=True,
                padding_value=self.llama_tokenizer.pad_token_id,
            )  # bsz x s1
            p_before_attn_mask = p_before_token_ids.ne(
                self.llama_tokenizer.pad_token_id
            )
        else:
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(
                self.device
            )  # [s1, s1...] list of batch size
            p_before_token_ids = p_before_tokens.input_ids.expand(
                batch_size, -1
            )  # bsz x s1
            p_before_attn_mask = p_before_tokens.attention_mask.expand(
                batch_size, -1
            )  # bsz x s1
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_token_ids
        )  # .expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(
            batch_size, -1, -1
        )  # bsz x s2 x embed_dim
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_token_ids.dtype,
                device=p_before_token_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(
            bos
        )  # bsz x 1 x embed_dim
        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+NumToken+s2) x embed_dim

        # make target ids for prefix part
        empty_targets = (
            torch.ones(
                [batch_size, 1 + p_before_embeds.size()[1] + self.num_vision_token],
                dtype=torch.long,
            )
            .to(self.device)
            .fill_(-100)  # 1 (bos) + s1 + num_image_tokens (image vector)
        )  # bsz x (1 + s1 + 1)
        targets = torch.cat(
            [empty_targets, target_ids], dim=1
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        # atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + self.num_vision_token], dtype=torch.long).to(self.device) # bsz x (1[bos] + s1 +num_image_tokens)
        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(
            self.device
        )  # bsz x 1
        atts_img = torch.ones([batch_size, self.num_vision_token], dtype=torch.long).to(
            self.device
        )  # bsz x num_image_tokens
        attention_mask = torch.cat(
            [atts_bos, p_before_attn_mask, atts_img, attention_mask], dim=1
        )
        assert (
            attention_mask.size() == targets.size()
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        return inputs_embeds, targets, attention_mask

    def catALL(self, detection_gt, max_obj, device,*args):
        vis_embed_list = []
        for i in detection_gt:
            vis_embeds, _ = self.encode_obj_pcl(device, i[:max_obj])
            vis_embeds = self.llama_proj(vis_embeds)
            vis_embed_list.append(vis_embeds)
        return vis_embed_list

    def chooseOne(self, detection_gt, max_obj, device, vision_paths,obj_points, *args):
        # vis_embed_list = []
        # for classobj in vision_paths:
        #     mesh_vertices = o3d.io.read_point_cloud(classobj)
        #     point_cloud = np.asarray(mesh_vertices.points)
        #     while (len(point_cloud) < 8192):
        #         newcut_point_cloud = point_cloud + 0.00001
        #         point_cloud = np.concatenate((point_cloud, newcut_point_cloud), axis=0)
        #     point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 8192, replace=False)]
        #     vis_embed_list.append(point_cloud)
        vis_embed_list = obj_points
        vis_embed_list, _ = self.encode_obj_pcl(device, vis_embed_list)
        vis_embed_list = self.llama_proj(vis_embed_list)
        return vis_embed_list

    def forward(self, inputs):
        # image_paths = inputs['image_paths']
        assert (
            self.vision_type == inputs["vision_type"]
        ), "{} expected but {} given".format(self.valid_type, inputs["vision_type"])
        task_type = inputs["task_type"]
        vision_paths = inputs["vision_paths"]
        points_path = inputs["points_path"]
        label_path = inputs["label_path"]
        max_obj = self.max_obj_len

        vis_embed_list,class_box_gt = [],[]
        points_path = points_path[0]
        #处理vis_embed_list,scene
        if task_type[0] in ['Classification3d','DescriptionObj3d','ConversationObj3d']:
            obj_numpy = np.load(points_path)
            x, y, z = np.mean(obj_numpy, axis=0)
            w, l, h = np.max(np.abs(obj_numpy), axis=0)
            class_box_gt = [[round(x, 1), round(y, 1), round(z, 1)]]
            # class_box_gt = [[round(x, 2), round(y, 2), round(z, 2), round(w, 2), round(l, 2), round(h, 2)]]

            # class_embed = np.load(points_path)
            class_embed, _ = self.encode_obj_pcl(self.device, [obj_numpy])
            class_embed = self.llama_proj(class_embed)
            vis_embed_list = class_embed
        else:
            vis_numpy_list = []
            for i in points_path['points']:
                obj_numpy = np.load(i)
                vis_numpy_list.append(obj_numpy)
                x, y, z = np.mean(obj_numpy, axis=0)
                w, l, h = np.max(np.abs(obj_numpy), axis=0)
                # class_box_gt.append([round(x, 2), round(y, 2), round(z, 2), round(w, 2), round(l, 2), round(h, 2)])
                class_box_gt.append([round(x, 1), round(y, 1), round(z, 1)])
            vis_embeds, _ = self.encode_obj_pcl(self.device, vis_numpy_list[:max_obj])
            vis_embeds = self.llama_proj(vis_embeds)
            vis_embed_list = vis_embeds



        # #处理vis_embed_list
        # if task_type[0] in options:
        #     vis_embed_list = options[task_type[0]](detection_gt, max_obj, self.device,vision_paths,torch.stack(obj_points))
        # else:
        #     vis_embed_list = []
        #     print("Wrong task_type,choose one from [Detection,Counting,Class,PositionRelation,VG,RoomDetection,Navigation]")

        # #处理无bbox的情况
        # class_gt = inputs["class_gt"]
        # class_box_gt = inputs["class_box_gt"]
        # if task_type[0] == "Classification3d":
        #     vis_embed_list = vis_embed_list
        #     class_gt = ['Unknown']*len(task_type)
        #     class_box_gt = [
        #         [round(value, 2) for value in torch.mean(box, dim=0).tolist()] +  # xyz均值
        #         [round(value, 2) for value in torch.max(torch.abs(box), dim=0).values.tolist()]  # wlh最大绝对值
        #         for box in inputs['obj_points']
        #     ]
        #     # class_box_gt = [[0,0,0,2,2,2]]*len(task_type)
        # batch_input_ids = []
        # index = 0
        # for c,b in zip(class_gt[:max_obj],class_box_gt[:max_obj]):
        #     class_name = c +str(b)+'!'
        #     class_name = self.llama_tokenizer(class_name, add_special_tokens=False).input_ids
        #     batch_input_ids.append(torch.LongTensor(class_name))
        #     index+=1
        #处理无bbox的情况

        batch_input_ids = []
        for index,b in enumerate(class_box_gt[:max_obj]):
            class_name = 'obj'+str(index)+str(b)+'!'
            class_name = self.llama_tokenizer(class_name, add_special_tokens=False).input_ids
            batch_input_ids.append(torch.LongTensor(class_name))

        input_ids = rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.llama_tokenizer.pad_token_id
        )
        input_ids = input_ids.to(self.device)
        input_embeds = self.llama_model.model.model.embed_tokens(input_ids)

        #插入vision embedings
        vision_embeds_my = []
        for index, vis in enumerate(vis_embed_list):
            vision_embeds_my_b = []
            vision_embeds_my_b.append(vis.unsqueeze(dim=0))
            vision_embeds_my_b.append(input_embeds[index])
            vision_embeds_my.append(torch.cat(vision_embeds_my_b))

        vision_embeds = torch.cat(vision_embeds_my).unsqueeze(dim=0)

        output_texts = inputs["output_texts"]
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len, self.vision_type
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            vision_embeds,
            input_ids,
            target_ids,
            attention_mask,
            self.use_system,
            task_type,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=not self.use_flash_attn,
        )
        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(
            torch.long
        )  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def extract_multimodal_feature(self, inputs):
        """Extract multimodal features from the input in Generation (Test)

        :param Dict inputs: input dict; modality: path
        :return _type_: _description_
        """
        features = []
        if 'image_paths' in inputs and inputs["image_paths"]:
            image_embeds, _ = self.encode_image(inputs["image_paths"])
            features.append(image_embeds)
        if 'images' in inputs and inputs["images"]:  # image objects input in testing
            image_embeds, _ = self.encode_image_object(inputs["images"])
            return image_embeds
        if "pcl_paths" in inputs and inputs["pcl_paths"]:
            # pcl_embeds, _ = self.encode_pcl(inputs["pcl_paths"])
            pcl_embeds, _ = self.test_encode_pcl(inputs["pcl_paths"],inputs["obj_list"][:self.max_obj_len],inputs['list_of_objpoints'])
            return pcl_embeds
            features.append(pcl_embeds)
        # TODO: Cautions HERE! Multimodality allowed in test ONLY!
        feature_embeds = (
            torch.cat(features).sum(dim=0).unsqueeze(0)
        )  # sum all modality features together
        return feature_embeds

    def prepare_generation_embedding(self, inputs):
        """prepare for generation

        :param class inputs: model
        :return Dict: generation input
        """
        eov = VISION_TAGS["eov"][self.vision_type]
        # TODO: add System header & image token size
        prompt_list = inputs["prompt"]  # questions from user
        if len(inputs["modality_embeds"]) == 1:
            feature_embeds = inputs["modality_embeds"][0]
        else:
            feature_embeds = self.extract_multimodal_feature(inputs)
            #inputs["modality_embeds"].append(feature_embeds)


        if inputs["task_type"] in ["Classification",'DescriptionObj3d','ConversationObj3d']:
            x,y,z = np.mean(inputs['list_of_objpoints'], axis=0)
            class_box_gt = [[round(x, 1), round(y, 1), round(z, 1)]]
            w,l,h = np.max(np.abs(inputs['list_of_objpoints']), axis=0)
        # class_box_gt = [[round(x, 2),round(y, 2),round(z, 2),round(w, 2),round(l, 2),round(h, 2)]]
        else:
            class_list = [classname["name"] for classname in inputs["obj_list"]]
            class_box_gt = [[round(classname['BoundingBox'][0], 1),round(classname['BoundingBox'][1], 1),round(classname['BoundingBox'][2], 1)] for classname in inputs["obj_list"]]

        max_obj = self.max_obj_len
        batch_input_ids,class_name_target_ids = [],[]

        for index,b in enumerate(class_box_gt[:max_obj]):
        #     class_name = 'obj'+str(index)+str(b)+'!'
            class_name = str(b)+'!'
            class_name = self.llama_tokenizer(class_name, add_special_tokens=False).input_ids
            class_name_target_ids += [-100] * len(class_name)
            batch_input_ids.append(torch.LongTensor(class_name))

        input_ids = rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.llama_tokenizer.pad_token_id
        )
        input_ids = input_ids.to(self.device)
        input_embeds = self.llama_model.model.embed_tokens(input_ids)

        #插入vision embedings
        vision_embeds_my = []
        for index, vis in enumerate(feature_embeds):
            vision_embeds_my.append(vis)
            vision_embeds_my.append(input_embeds[index])
        vision_embeds = torch.cat(vision_embeds_my).unsqueeze(dim=0)


        batch_size = vision_embeds.shape[0]
        p_before = make_prompt_start(
            vision_type=self.vision_type
        )  # no system header in test
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.llama_model.model.embed_tokens(
            p_before_tokens.input_ids
        ).expand(
            batch_size, -1, -1
        )  # bsz x s1 x embed_dim

        p_after_texts = [f"{eov} " + prompt + "\n### Assistant:" for prompt in prompt_list]
        p_after_tokens = self.llama_tokenizer(
            p_after_texts,
            padding="longest", return_length=True, # padding right
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
        p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)

        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_tokens.input_ids.dtype,
                device=p_before_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.embed_tokens(
            bos
        )  # bsz x 1 x embed_dim

        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, vision_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+NumVisionToken+s2) x embed_dim

        # p_after_embeds are on right, so the pads are right,
        # we need to move all inputs_embeds to right,
        # to make the pads on left
        tokens_len = inputs_embeds.shape[1] - p_after_masks_len
        new_inputs_embeds = torch.zeros_like(inputs_embeds)
        inputs_embeds_masks = torch.zeros(inputs_embeds.shape[:-1],
                                         dtype=torch.int64, device=self.device)
        for idx in range(batch_size):
            inputs_embeds_masks[idx, -tokens_len[idx]:] = 1
            new_inputs_embeds[idx, -tokens_len[idx]:, :] = inputs_embeds[idx, :tokens_len[idx], :]
            new_inputs_embeds[idx, :-tokens_len[idx], :] = inputs_embeds[idx, tokens_len[idx]:, :]

        return new_inputs_embeds, inputs_embeds_masks

    def generate(self, inputs):
        """
        inputs = {
            'image_paths': optional,
            'mode': generation mode,
            'prompt': human input prompt,
            'max_tgt_len': generation length,
            'top_p': top_p,
            'temperature': temperature
            'modality_embeds': None or torch.tensor
            'modality_cache': save the image cache
        }
        """
        input_embeds, input_masks = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList(
            [LAMMStoppingCriteria([[2277, 29937], [835]], input_embeds)]
        )
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            max_new_tokens=inputs["max_tgt_len"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return output_text
