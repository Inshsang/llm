import os
from model.openlamm import LAMMPEFTModel
import torch
import json,jsonlines
import argparse
from conversations import conv_templates
from tqdm import tqdm
from bigmodelvis import Visualization
from datasets import load_3Deval_dataset
import pickle
answers_file = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type", type=str, default='VisualGrounding_plus', help="task type" #Detection,Counting,Classification,PositionRelation,
                                                                            # VisualGrounding,RoomDetection,Navigation,
                                                                                #VQA,Relation,Caption
    )
    # parser.add_argument("--dataset-name", default="detection")
    parser.add_argument(
        "--choose", type=bool, default=True, help="choose scene of objects <= 12"
    )
    parser.add_argument('--model', type=str, default='lamm_peft')
    parser.add_argument(
        "--encoder_pretrain",
        type=str,
        default="epcl",
        choices=("clip", "epcl"),
        help="Vision Pretrain Model",
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        default="/media/kou/Data3/htc/epcl_scannet_vit-L-14_256tokens_latest.pth",
        help="path of vision pretrained model; CLIP use default path in cache",
    )
    parser.add_argument(
        "--vicuna_ckpt_path",
        type=str,
        # required=True,
        default="/media/kou/Data3/htc/vicuna-7b/",
        help="path of LLM, default: Vicuna",
    )
    parser.add_argument(
        "--train_stage", type=int, default=2, help="1，2for obj alignment；3 for all"
    )
    parser.add_argument(
        "--delta_ckpt_path",
        type=str,
        default="/media/kou/Data1/htc/LAMM/ckpt/--ALLNavi_test/pytorch_model_ep1.pt",
        # default="/media/kou/Data1/htc/LAMM/ckpt/--ALLV0/pytorch_model_ep2.pt",
        # default="/media/kou/Data1/htc/LAMM/ckpt/--ALL/pytorch_model_ep1.pt",
        help="path of delta parameters from previous stage; Only matter for stage 2",
    )
    parser.add_argument('--stage', type=int, default=2,)
    # LoRA configurations
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    # Embedding configurations
    parser.add_argument('--vision_feature_type', type=str, default='local', choices=('local', 'global'))
    parser.add_argument('--vision_output_layer', type=int, default=-2, choices=(-1, -2), help='the layer to output visual features; -1 means global from last layer')
    parser.add_argument('--num_vision_token', type=int, default=256) # the maximum sequence length
    # Test configurations
    parser.add_argument(
        "--max_obj_len", type=int, default=30, help="Root dir for images"
    )
    parser.add_argument('--max_tgt_len', type=int, default=1200, help="maximum length of target sequence at least 400; in case of 1 vision token")
    parser.add_argument('--conv_mode', type=str, default='simple')
    parser.add_argument("--inference-mode", default='common')
    parser.add_argument("--bs", type=int,default=1)
    parser.add_argument("--base-data-path", default="/media/kou/Data1/htc/LAMM/data")
    parser.add_argument("--answers-dir", default="../answers")
    # parser.add_argument("--dataset-name", required=True)
    # parser.add_argument("--base-data-path", required=True)
    # parser.add_argument("--answers-dir", required=True)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.vision_feature_type == 'local':
        args.vision_output_layer = -2
        args.num_vision_token = 256
    elif args.vision_feature_type == 'global':
        args.vision_output_layer = -1
        args.num_vision_token = 1
    else:
        raise NotImplementedError('NOT implement vision feature type: {}'.format(args.vision_feature_type))
    
    assert os.path.exists(args.delta_ckpt_path), "delta checkpoint not exists!"
    assert os.path.exists(args.vicuna_ckpt_path), "vicuna checkpoint not exists!"
    assert os.path.exists(args.encoder_ckpt_path), "vision encoder checkpoint not exists!"
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    return args


single_infernce_dataset = [
    'ScanNet', 'ScanRefer', 'ScanQA_multiplechoice'
]


def generate_conversation_text(args, input_list, history, sys_msg=None):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    conv = conv_templates[args.conv_mode]
    if sys_msg:
        conv.system = sys_msg
    prompts_list = []
    for input in input_list:
        prompts = ''
        prompts += conv.system 
        for q, a in history:
            prompts += "{} {}: {}\n{} {}: {}\n".format(conv.sep, conv.roles[0], q, conv.sep, conv.roles[1], a)
        prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], input)
        prompts_list.append(prompts)
    return prompts_list


def predict(
    args,
    model,
    input,
    pcl_paths, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    sys_msg,
    obj_list,
    list_of_objpoints,
    task_type,
):
    prompt_text = generate_conversation_text(args, input, history, sys_msg)
    response = model.generate({
        'prompt': prompt_text,
        'pcl_paths': pcl_paths,
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': [],
        'obj_list': obj_list,
        'list_of_objpoints':list_of_objpoints[:args.max_obj],
        'task_type':task_type
    })
    history.append((input, response))
    return history


def Class_response(args,
                     model,
                     input,
                     pcl_paths,
                     sys_msg,
                     obj_lists,
                     list_of_objpoints):
    """get response text by default

    :param args args: input arguments
    :param model model: model class
    :param input input: input text
    :param object pcl_paths: image objects
    :param str sys_msg: system message for test
    :return list: list of response
    """
    pcl_paths[0] = "/media/kou/Data3/htc/Objects_npy/" + pcl_paths[0][39:]
    history = predict(
        args=args,
        model=model,
        input=input,
        pcl_paths=pcl_paths,
        max_length=20,
        top_p=0.95,
        temperature=0.8,
        history=[],
        sys_msg=sys_msg,
        obj_list=obj_lists,
        list_of_objpoints=list_of_objpoints,
        task_type=args.task_type,
    )
    response = history[-1][1]
    ans_list = []

    for res in response:
        ans_list.append(res.split('###')[0])
    return ans_list

def Other_response(args,
                    model,
                    input,
                    pcl_paths,
                    sys_msg,
                    obj_lists,
                    list_of_objpoints):
    """get response text by default

    :param args args: input arguments
    :param model model: model class
    :param input input: input text
    :param object pcl_paths: image objects
    :param str sys_msg: system message for test
    :return list: list of response
    """
    src_id = pcl_paths[0][37:-4]
    history = predict(
        args=args,
        model=model,
        input=input,
        pcl_paths=pcl_paths,
        max_length=args.max_tgt_len,
        top_p=0.9,
        temperature=0.8,    #0.8,1,2
        history=[],
        sys_msg=sys_msg,
        obj_list=obj_lists[src_id],
        list_of_objpoints=list_of_objpoints,
        task_type=args.task_type,
    )
    response = history[-1][1]
    ans_list = []
    
    for res in response:
        ans_list.append(res.split('###')[0])
    return ans_list


def Detection_response(args,
                       model,
                       input,
                       pcl_paths,
                       sys_msg,
                       obj_lists,
                       list_of_objpoints):
    """get response text by default

    :param args args: input arguments
    :param model model: model class
    :param input input: input text
    :param object pcl_paths: image objects
    :param str sys_msg: system message for test
    :return list: list of response
    """
    src_id = pcl_paths[0][37:-4]
    history = predict(
        args=args,
        model=model,
        input=input,
        pcl_paths=pcl_paths,
        max_length=args.max_tgt_len,
        top_p=0.9,
        temperature=0.8,
        history=[],
        sys_msg=sys_msg,
        obj_list=obj_lists,
        list_of_objpoints=list_of_objpoints,
        task_type=args.task_type,
    )
    response = history[-1][1]
    ans_list = []

    for res in response:
        ans_list.append(res.split('###')[0])
    return ans_list


def main(args):
    # load model
    model = LAMMPEFTModel(**args.__dict__)
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    print(f'[!] merging LoRA weights ...')
    model.llama_model = model.llama_model.merge_and_unload()
    model = model.eval().half().cuda()
    Visualization(model).structure_graph()
    print(f'[!] init the LLM over ...')
    
    # load data

    inference_mode = args.inference_mode
    batch_size = args.bs
    dataloader = load_3Deval_dataset(
        args.base_data_path,
        args.task_type,
        inference_mode,
        batch_size = batch_size
    )
    sys_msg = dataloader.dataset.system_msg
    task_name = dataloader.dataset.task_type

    answers_file_name = task_name + '.json'
    answers_file = os.path.join(args.answers_dir, answers_file_name)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # exist_list = []
    # existing = jsonlines.Reader(open("/media/kou/Data1/htc/LAMM/answers/Classification.jsonl"))
    # for e in existing:
    #     exist_list.append(e["pcl"][0])

    with open("/media/kou/Data3/htc/dataset/Object/my_test_8192pts_fps.dat", 'rb') as f:
        list_of_objpoints = pickle.load(f)
    list_of_class_name = json.load(open("/media/kou/Data3/htc/dataset/Object/my_test.json"))
    #删除只有一个点的物体
    list_of_objpoints[0] = [npy for index, npy in enumerate(list_of_objpoints[0]) if index != 773]
    list_of_objpoints[1] = [npy for index,npy in enumerate(list_of_objpoints[1]) if index!=773]

    if task_name in ['Classification','DescriptionObj3d','ConversationObj3d']:
        args.max_obj = 12
        obj_lists = [{'name':'Unknown','BoundingBox':[0,0,0,2,2,2]}]
    elif task_name == ["Detection"]:
        args.max_obj = 30
        Detection_lists = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Test/Detection.json"))
    else:
        args.max_obj = 30
        obj_lists = json.load(open("/media/kou/Data1/htc/LAMM/data/metadata/" + "Detection" + ".json", 'r'))


    ans_list = []
    ans_file = open(os.path.splitext(answers_file)[0] + '.jsonl', 'w')
    for index,data_item in enumerate(tqdm(dataloader)):
        # if data_item["pcl"][0] in exist_list:
        #     continue
        prompt = data_item['query']
        pcl_paths = data_item['pcl']

        if task_name == 'Detection':
            obj_lists = Detection_lists[index]['object']
            response_func = Detection_response
        elif task_name in ['Classification','DescriptionObj','ConversationObj']:
            response_func = Class_response
        else:
            response_func = Other_response

        if task_name == 'Relation':
            prompt = ["Analyze the relationships between various objects."]
        elif task_name == 'Caption':
            prompt = ["Write a detailed caption by classifying and describing different rooms in 150-200 words, illustrating their types, appearance and other information such as functionalities, usages, daily-life knowledge."]

        if task_name == 'VQA':
            # prompt = ["Generate 5 Q&As about the objects in scene. All conversations must be logically related."
            #           "Format your responses as Q1,A1; Q2,A2; Q3,A3; Q4,A4; Q5,A5. "
            #           ]
            prompt = ["You need to create three question-and-answer pairs centered around the objects, ensuring that the context is interconnected. Format your response as a list,[Q1,A1,Q2,A2,Q3,A3]. Response must be logically related."
                      ]
            answer_list = response_func(
                args=args,
                model=model,
                input=prompt,
                pcl_paths=pcl_paths,
                sys_msg=sys_msg,
                obj_lists=obj_lists,
                list_of_objpoints=list_of_objpoints[0][index]
            )
            # for i in range(4):
            #     answer_list = response_func(
            #         args=args,
            #         model=model,
            #         input=prompt,
            #         pcl_paths=pcl_paths,
            #         sys_msg=sys_msg,
            #         obj_lists=obj_lists,
            #         list_of_objpoints=list_of_objpoints[0][index]
            #     )
            #     prompt = [prompt[0]+answer_list[0]]
        else:
            answer_list = response_func(
                args=args,
                model=model,
                input=prompt,
                pcl_paths=pcl_paths,
                sys_msg=sys_msg,
                obj_lists=obj_lists,
                list_of_objpoints = list_of_objpoints[0][index]
            )

        for id, output in zip(data_item['id'], answer_list):
            #print(output,prompt[0])
            ans_dict = {"id": id,
                        "pcl": data_item['pcl'],
                        "text": output,
                        "delta_path": args.delta_ckpt_path
                        }
            ans_list.append(ans_dict)
            ans_file.write(json.dumps(ans_dict) + "\n")
            ans_file.flush()

    ans_file.close()
    # dump all
    ans_file = open(answers_file, "w")
    ans_file.write(json.dumps(ans_list, indent=4))
    ans_file.flush()
    ans_file.close()
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
