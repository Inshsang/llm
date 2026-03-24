# Agent Guide for `Project/llm`

## 1. 项目定位

这是一个基于 LAMM 的多模态大模型项目，核心目标是让 LLM 同时处理：

- 2D 图像
- 3D 点云
- 指令微调、评测与推理

当前仓库不是纯净上游代码，已经包含本地定制内容，尤其偏向 3D 点云任务和 agent-style 推理流程。后续改动时，应优先按“现有脚本和默认路径”理解项目，而不是只按上游 README 理解。

## 2. 代码结构

建议先从这些路径建立上下文：

- `README.md`: 项目总览，说明这是 LAMM 框架与数据集仓库。
- `docs/lamm.md`: 更简洁的 LAMM 文档页。
- `src/train.py`: 主训练入口。
- `src/inference_2d.py`: 2D 推理入口。
- `src/inference_3d.py`: 3D 推理入口。
- `src/agent_inference.py`: 当前仓库里最重要的 3D agent 推理脚本，包含任务路由、proposal 复用、反思修正等逻辑。
- `llm.py`: 仓库根目录的本地快捷推理脚本，不属于典型上游布局，但应视为当前工作流的一部分。
- `src/model/openlamm.py`: 核心模型装配逻辑，负责视觉编码器、Vicuna/LLaMA、LoRA、生成输入拼接。
- `src/model/training_agent.py` 与 `src/model/agent.py`: DeepSpeed 训练封装。
- `src/config/train.yaml` / `src/config/train_ds3.yaml`: 训练配置。
- `src/datasets/`: 数据集加载与 system prompt 等逻辑。
- `src/tools/ChEF/`: 评测脚本。
- `src/scripts/` 与 `src/tools/LAMM/`: 训练、推理、评测 shell 脚本。

## 3. 默认技术栈

- Python 3.10
- PyTorch 1.12.x
- DeepSpeed 0.9.2
- Transformers 4.29.1
- PEFT 0.3.0
- timm / decord / nltk / trimesh / plyfile
- 可选加速：
  - FlashAttention
  - xFormers

`requirements.txt` 中还包含 3D 相关依赖；如果涉及 EPCL/PointNet2，注意有额外编译步骤。

## 4. 运行入口与真实默认值

### 4.1 训练

主入口：

- `python src/train.py`

当前仓库里的默认参数不是通用示例，而是本地环境绑定值，例如：

- 默认配置：`src/config/train_ds3.yaml`
- 默认数据：`/data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/WholeTrain/Agent_v1_demo.json`
- 默认视觉类型：`pcl`
- 默认视觉编码器：`epcl`
- 默认 Vicuna 路径：`/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0/`
- 默认 EPCL 权重：`/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth`

如果要改训练脚本，先确认这些绝对路径是否仍有效，不要直接假设仓库可脱离 `/data/HTC` 独立运行。

### 4.2 推理

常用入口：

- `python src/inference_2d.py`
- `python src/inference_3d.py`
- `python src/cli_demo.py`
- `python src/agent_inference.py`
- `python llm.py`

其中：

- `src/inference_3d.py` 更接近传统 LAMM 3D 推理。
- `src/agent_inference.py` 是当前仓库最值得优先理解的脚本，面向 3D agent 任务。
- `llm.py` 是一个本地封装过的快速交互脚本，重点用于点云分类等场景，提示词、history 处理和输出清洗都做过定制。

## 5. 模型与任务理解

`src/model/openlamm.py` 是主要事实来源。阅读和修改时重点关注：

- `encoder_pretrain`: 目前主要支持 `clip` 与 `epcl`
- `vision_type`: `image` 或 `pcl`
- `train_stage`: 不同 stage 对 LoRA / 参数冻结 / checkpoint 加载方式有影响
- `vision_feature_type`: `local` 与 `global` 会影响 `num_vision_token` 和输出层

当前项目明显偏向 3D 点云任务，尤其包括：

- Classification
- Detection
- Counting
- RoomDetection
- PositionRelation
- VisualGrounding_plus

这些任务在 `src/agent_inference.py` 中有更直接的实现痕迹，应优先以该文件为准。

## 6. 数据与路径约定

仓库外部依赖很重，很多脚本默认读取 `/data/HTC/Data/...` 下的数据和模型。实际工作时默认假设以下几类资源存在：

- 点云数据：`/data/HTC/Data/dataset/object_1024_npy`
- benchmark / task json：`/data/HTC/Data/dataset/Benchmark/...`
- Vicuna 权重：`/data/HTC/Data/model_zoo/vicuna-7b/...`
- EPCL 权重：`/data/HTC/Data/model_zoo/epcl_ckpt/...`
- 训练输出：`/data/HTC/Data/model_zoo/llm_exe/...`

不要轻易把绝对路径批量改成相对路径，除非同时梳理所有脚本、配置和调用链。

## 7. 修改优先级建议

处理需求时，优先按下面顺序定位：

1. 若是训练问题，先看 `src/train.py`、`src/config/*.yaml`、`src/model/training_agent.py`
2. 若是 3D 推理问题，先看 `src/agent_inference.py` 和 `src/inference_3d.py`
3. 若是模型输入输出异常，直接看 `src/model/openlamm.py`
4. 若是 prompt / 对话模板问题，检查 `src/conversations.py` 和 `src/datasets/system_msg*`
5. 若是评测问题，检查 `src/eval.py`、`src/common_eval_*`、`src/tools/ChEF/*`、`src/ablation_stats.py`

## 8. 本仓库的本地定制特征

和标准学术仓库相比，这里有几类明显的本地改造，改动时不能忽略：

- 根目录新增 `llm.py`
- `src/agent_inference.py` 已经是大体量定制脚本，不是简单 demo
- 训练与推理脚本大量依赖绝对路径
- `src/train.py` 默认数据与保存目录已替换为本地实验路径
- 仓库中同时存在上游组件副本，如 `lightllm`、`Point-BERT`、`flash-attention` 等相关目录，但当前 `Project/llm` 的直接执行入口主要还是 `src/`

因此，修改前先判断需求究竟属于：

- 上游 LAMM 主流程
- 本地 3D agent 扩展
- 实验脚本/一次性分析脚本

不要把三类代码混在一起重构。

## 9. 编码与改动原则

- 优先做小步、可验证的改动，避免大规模整理 import 或重命名。
- 修改默认参数前，先检查是否会影响 `/data/HTC` 下已有实验流程。
- 如果改 `openlamm.py`，同步检查训练、2D 推理、3D 推理、agent 推理四条链路。
- 如果改 prompt 组装逻辑，检查：
  - system prompt
  - history 拼接
  - stop token / `###` 截断
  - task_type 是否和训练模板一致
- 如果改点云输入逻辑，检查：
  - `obj_list`
  - `list_of_objpoints`
  - bbox / proposal 复用格式
  - `.npy` 读入后 shape 与 dtype

## 10. 验证建议

做完改动后，至少按改动类型做最小验证：

- 训练相关：
  - 能否正常解析 `src/config/train_ds3.yaml`
  - `python src/train.py --help`
- 2D 推理相关：
  - `python src/inference_2d.py --help`
- 3D 推理相关：
  - `python src/inference_3d.py --help`
  - `python src/agent_inference.py --help`
- 本地快捷推理相关：
  - `python llm.py --help`

如果环境允许，再做一次带真实 checkpoint 的单样本 smoke test。

## 11. 对后续 Agent 的一句话建议

这个项目表面上是 LAMM，多数实际需求却会落到“本地定制的 3D 点云 agent 推理与训练链路”上。遇到行为不一致时，优先相信当前仓库中的脚本默认值和本地调用路径，而不是只相信上游文档。
