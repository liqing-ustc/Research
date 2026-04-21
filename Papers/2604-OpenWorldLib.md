---
title: "OpenWorldLib: A Unified Codebase and Definition of Advanced World Models"
authors: [DataFlow Team, Bohan Zeng, Daili Hua, Kaixin Zhu, Yifan Dai, Bozhou Li, Yuran Wang, Chengzhuo Tong, Yifan Yang, Mingkun Chang, Jianbin Zhao, Zhou Liu, Hao Liang, Xiaochen Ma, Ruichuan An, Junbo Niu, Zimo Meng, Tianyi Bai, Meiyi Qiang, Huanyao Zhang, Zhiyou Xiao, Tianyu Guo, Qinhan Yu, Runhao Zhao, Zhengpin Li, Xinyi Huang, Yisheng Pan, Yiwen Tang, Yang Shi, Yue Ding, Xinlong Chen, Hongcheng Gao, Minglei Shi, Jialong Wu, Zekun Wang, Yuanxing Zhang, Xintao Wang, Pengfei Wan, Yiren Song, Mike Zheng Shou, Wentao Zhang]
institutes: [Peking University, Kling Team Kuaishou Technology, Zhongguancun Academy, HKUST, Tsinghua University, NUS, SJTU, SYSU]
date_publish: 2026-04-06
venue: arXiv
tags: [world-model, VLA, 3D-representation]
paper: https://arxiv.org/abs/2604.04707
website: 
github: https://github.com/OpenDCAI/OpenWorldLib
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] OpenWorldLib: A Unified Codebase and Definition of Advanced World Models
> - **核心**: 给 "world model" 一个可操作的定义——以 perception 为中心、具备 interaction 与 long-term memory 的 framework；并配套提供一个把 video generation / 3D reconstruction / multimodal reasoning / VLA 拼装到统一 inference pipeline 里的代码库
> - **方法**: 五模块抽象（Operator / Synthesis / Reasoning / Representation / Memory）+ Pipeline 调度，每个模块给出 Base 模板类；下游模型通过继承 + `from_pretrained` 接入
> - **结果**: 在 A800/H200 上集成并定性评测了 Matrix-Game-2 / Hunyuan-GameCraft / Hunyuan-WorldPlay / Lingbot-World / YUME-1.5 / Wan-IT2V / WoW / Cosmos / VGGT / InfiniteVGGT / FlashWorld / π0 / π0.5 / LingBot-VA 等代表方法，无定量 benchmark 数字
> - **Sources**: [paper](https://arxiv.org/abs/2604.04707) | [github](https://github.com/OpenDCAI/OpenWorldLib)
> - **Rating**: 2 - Frontier（Position + tooling 论文；定义框架对 world model 概念争论提供 reference point，codebase 覆盖当前主流集成方向，但科学贡献集中在定义+wrapper 层，非奠基性）

**Key Takeaways:**
1. **Position 性贡献为主**: 论文核心是 "definition + taxonomy"，而非新模型；它明确把 text-to-video（包括把 Sora 称作 world simulator 的说法）、code generation、avatar video 排除出 world model 范畴
2. **Definition**: world model = 以 perception 为中心、具备 interaction 与 long-term memory 能力的 model/framework，用来理解和预测复杂世界——强调能力层而非任务层或架构层
3. **OpenWorldLib codebase**: 五个 Base 模板类（Operator / Synthesis / Reasoning / Representation / Memory）+ Pipeline 编排，类似 HuggingFace `transformers` 风格的 wrapper layer，把异构 world-model 候选模型塞进同一调用接口
4. **评测无量化结果**: §5 全是 "Method A 比 Method B 视觉质量更好" 这类 qualitative claim，没有数值 benchmark；可重现性主要靠开源代码

**Teaser. OpenWorldLib 整体框架示意。**
![](https://arxiv.org/html/2604.04707v1/x3.png)

---

## 1. Definition: 什么算 World Model

论文先回到经典 world model 的三个条件分布：

$$
p(s_{t+1} \mid s_t, a_t), \quad p(o_t \mid s_t), \quad r_t \sim p(r_t \mid s_t, a_t)
$$

其中 $s_t$ 是 latent state（隐含 long-horizon memory）、$a_t$ 是 action（被泛化到包括 generation / manipulation 等输出）、$o_t$ 是观测、$r_t$ 是交互奖励。作者指出**很多任务形式上满足这三个分布，但其实没在做 world model 该做的事**。所以需要从 "core objective" 重新切：

> a world model is a model or framework centered on building internal representations from perception, equipped with action-conditioned simulation and long-term memory capabilities, for understanding and predicting the dynamics of a complex world.

这个定义有意识地**绕开架构和任务**——next-frame prediction 只是一个实现形式，不是定义。

### 1.1 算 World Model 的任务

- **Interactive Video Generation**: next-frame prediction 仍是最被认可的 paradigm，从 regression-based 模型到 diffusion，再到 game video / camera-controlled video，并被嵌入 [[2410-Pi0|π0]] 类 VLA 与自动驾驶
- **Multimodal Reasoning**: spatial / omni / temporal / causal reasoning，外加新的 latent reasoning（绕开 text-centric pre-training，直接处理高维连续真实世界信号）
- **Vision-Language-Action (VLA)**: 两条路线——MLLM 直接预测 action，或者把 action prediction 与 video generation 联合（用 future frame prediction 做 action planning）
- **3D & Simulator**: 3D reconstruction（VGGT / InfiniteVGGT / OmniVGGT 等）提供 verifiable 的物理空间；simulator（[[2501-Cosmos|Cosmos]]、FlashWorld、Hunyuan 系列）做 sandbox

### 1.2 不算 World Model 的任务

这一节是论文最有"立场"的部分：

- **Text-to-Video Generation**（含 Sora）: 缺乏复杂的 perceptual input，输出 video 不等于理解物理规则
- **Code generation / Web search**: 借用了 long-term interaction 的结构，但没有 multimodal physical input
- **Avatar video generation**: 即便是 multimodal + long-term interaction，主要服务娱乐场景，不涉及对复杂物理世界的理解

> 这个划分其实暗含一个判据：**输入端是否包含来自物理世界的复杂多模态感知**。比 "next-frame prediction 算不算" 这种架构层判据更清晰，但仍然是定性的——例如 "对物理世界的理解" 本身就难以操作化。

---

## 2. OpenWorldLib Framework

**Figure 2. 五模块 + Pipeline 整体架构。**
![](https://arxiv.org/html/2604.04707v1/x4.png)

**Figure 3. Implicit vs. Explicit representation 视角。**
![](https://arxiv.org/html/2604.04707v1/x5.png)

整个框架围绕五个模块 + 一个 Pipeline。每个模块都给出一个 `Base*` 抽象类，下游具体模型继承实现。

### 2.1 Operator —— 输入归一化

把异构的 raw input（text / image / continuous control / audio）转成下游模型能吃的 standardized tensor。两个职责：**Validation**（shape / type 校验）+ **Preprocessing**（resize / tokenize / normalize action space）。模板：

```python
class BaseOperator:
    def __init__(self):
        self.current_interaction = []
        self.interaction_template = []
    def get_interaction(self, interaction_list): ...
    def check_interaction(self, interaction): ...
    def process_interaction(self): raise NotImplementedError
    def process_perception(self): raise NotImplementedError
```

### 2.2 Synthesis —— Implicit 生成

负责 visual / audio / 其他 physical signal 的生成。三个子层：

- **Visual Synthesis**: 文本/参考图/场景规格 → frame tensor / decoded clip / API asset。组合 text encoder + latent decoder + diffusion or flow core + scheduler
- **Audio Synthesis**: 文本（+ 可选 video feature）→ waveform，需要 sampling rate、duration、guidance、step budget 等 user-facing knobs
- **Other Signal Synthesis (VLA)**: policy 初始化 + action space 对齐（discrete token ↔ continuous kinematic state），再做 context-conditioned action synthesis

模板核心是 `from_pretrained()` + `api_init()` + `@torch.no_grad() predict()`，明显借鉴 HuggingFace 风格。

### 2.3 Reasoning —— 多模态理解

分三类：

- **General Reasoning**: 通用 MLLM（处理 text / image / audio / video）
- **Spatial Reasoning**: 3D 空间理解、object localization
- **Audio Reasoning**: 音频信号理解

接口与 Synthesis 镜像：`from_pretrained()` + `api_init()` + `@torch.no_grad() inference()`。

### 2.4 Representation —— Explicit 3D

显式表示模块，与 Synthesis 的 implicit 表示并列：

- **3D Reconstruction**: input → point cloud / depth map / camera pose
- **Simulation Support**: 提供 manual environment 让 world model 验证预测 action 的物理一致性
- **Service Integration**: local inference + cloud API，可导出到外部 physics engine

接口：`from_pretrained()` + `api_init()` + `get_representation(data)`。

### 2.5 Memory —— 长期上下文

跨 turn 记录文本 / 视觉 feature / action 轨迹 / scene state，提供 context retrieval 与 session 隔离。四个核心方法：

```python
class BaseMemory:
    def record(self, data, metadata=None, **kwargs): ...     # 写入
    def select(self, context_query, **kwargs): ...           # 检索
    def compress(self, memory_items, **kwargs): ...          # 压缩
    def manage(self, **kwargs): ...                          # 生命周期
```

> ❓ Memory 模块的接口定义是最薄的——`record / select / compress / manage` 全是 `pass`，没有给出 reference implementation 或 retrieval strategy 的指引。在 long-horizon 交互里 memory 几乎是最难的部分，这层 API 是否真能 carry over 不同模型的 memory 设计存疑。

### 2.6 Pipeline —— 顶层调度

把 Operator → Memory query → Reasoning/Synthesis/Representation → Memory write 串起来。统一暴露：

- `from_pretrained()` 一键加载所有子模块
- `__call__()` 单轮 forward
- `stream()` 多轮 stateful 交互（自动读写 memory）

所有 task pipeline 继承 `BasePipeline`。本质上和 HuggingFace 的 `pipeline()` 抽象高度同构，只是把 memory 显式提到一等公民。

---

## 3. Discussion: 未来方向

论文 §4 给出几个判断：

1. **VLM 可能就是 world model 的合适基座**: 引用 Bagel 在 Qwen 架构上同时做到 multimodal reasoning 与 multimodal generation，作者认为 LLM-pretrained-on-internet-data 已经具备 world model 所需能力的雏形——所以**应该先把功能拼齐，再考虑专门架构**
2. **next-frame prediction 比 next-token prediction 信息量更大**，但效率问题需要硬件级改进——当前 byte 组织方式天然偏向 token，frame 在底层仍然以 token 处理
3. **Data-centric 方法会越发重要**: multimodal data synthesis、domain-specific augmentation、dynamic training、训练数据质量评估

> 这部分是 position paper 风格——结论方向感强但缺乏证据支撑。"硬件需要进化" 的论断尤其没有具体路径。

---

## 4. Evaluation

**实验设置**: NVIDIA A800 (80GB) + H200 (141GB)。**没有定量 benchmark 数字**，所有结论是 qualitative observation。

### 4.1 Interactive Video Generation

**Figure 4. Navigation / Interactive video generation 各方法对比。**
![](https://arxiv.org/html/2604.04707v1/x6.png)

- **Navigation video**: Matrix-Game-2 速度快但 long-horizon 有 color shifting；Lingbot-World、Hunyuan-GameCraft、YUME-1.5 质量较好；**Hunyuan-WorldPlay** 综合视觉效果最好
- **Interactive video**: Wan-IT2V 能做基本交互但物理一致性差；WoW 功能多但质量不如 [[2501-Cosmos|Cosmos]]

### 4.2 Multimodal Reasoning

涵盖 spatial reasoning（几何 / 布局问答、物体关系、step-by-step 空间推理）和 omni reasoning（混合模态 instruction following）。input：instruction + 可选 perceptual signal；output：自然语言（omni 设置下可能附带生成 audio）。

**Figure 5. Reasoning 任务示例。**
![](https://arxiv.org/html/2604.04707v1/x7.png)

### 4.3 3D Generation

**Figure 6. 3D reconstruction 各方法对比。**
![](https://arxiv.org/html/2604.04707v1/x8.png)

- VGGT / InfiniteVGGT 在大幅 camera 移动下出现几何不一致和 texture blurring
- FlashWorld 速度快但形状稳定与细节锐度难以兼顾
- 总体认为 3D generation 仍是 world model 的关键基础

### 4.4 VLA Generation

集成两套 simulation 评测：

- **AI2-THOR** for embodied video generation（photorealistic 场景渲染）
- **LIBERO** for VLA evaluation（可复现的物理 grounded manipulation）

集成的 VLA 方法包括 [[2410-Pi0|π0]]、[[2504-Pi05|π0.5]]（PaliGemma backbone + MoE action head）、LingBot-VA（video diffusion 联合建模 future frame + continuous action）。

> 整个评测部分定位更像是 "showcase 已集成方法" 而非严格 benchmark——没有跨方法的统一 metric、没有任务成功率数字、没有 latency / throughput 对比。

---
## 关联工作

### 基于
- [[2411-WorldModelSurvey|World Model Survey]]: 作者引用其论证 Sora 不构成完整 world simulator 的论点
- HuggingFace `transformers.pipeline`: 编程模型的明显前驱（虽未在论文中点名）

### 对比 / 集成
- [[2501-Cosmos|Cosmos]]: 论文集成的 interactive video generation 方法，被作为 high-quality baseline
- [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]]: 集成的 VLA 代表，建立在 PaliGemma + MoE action head 上
- Hunyuan-GameCraft / Hunyuan-WorldPlay / Matrix-Game-2 / YUME-1.5 / Lingbot-World / Wan-IT2V / WoW: 集成的 video generation 方法
- VGGT / InfiniteVGGT / OmniVGGT / FlashWorld: 集成的 3D generation 方法
- Bagel: §4 用作 "VLM 可作为 world model 基座" 的存在证明

### 方法相关
- [[2402-Genie|Genie]] 系列：interactive video generation / world model 的代表方向
- AI2-THOR / LIBERO: 用作 simulation-based evaluation 的 testbed

---
## 论文点评

### Strengths

1. **Definition 切角清楚**: 从 "core objective" 而非 "architecture / task" 切，把 next-frame prediction 显式降级为 "implementation form 之一" 而非定义；这个澄清对 field 内 "Sora 是不是 world simulator" 这种持续争论有价值
2. **Codebase 工程化**: 把目前散落的 video gen / 3D / VLA / reasoning 模型塞进同一调用接口，对想做 world model 系统级实验的研究者降低了起步成本——尤其是 multi-component pipeline 的 stitching
3. **明确划线**: 把 text-to-video / code generation / avatar video 显式排除，给社区一个可争论的 reference point。Position paper 的价值就在于 "可被 falsify"

### Weaknesses

1. **没有量化评测**: §5 全是定性判断（"A 比 B 质量更好"），没有 LIBERO 成功率、navigation accuracy、reasoning benchmark 分数等数字。对一个声称 "standardized evaluation pipeline" 的 framework 来说，这是核心缺口
2. **Memory 模块流于占位**: 接口最薄、所有方法都是 `pass`，没有 reference implementation 也没有 retrieval/compression 策略指引。但 long-horizon memory 才是 world model 真正的难点
3. **Definition 的可操作性边界模糊**: "对复杂物理世界的理解" 这种判据本身就难以操作化——例如 latent reasoning 算 "理解物理世界" 吗？避免用架构判据的代价是判据本身变得模糊
4. **本质上是 wrapper layer**: 类似 HuggingFace `transformers.pipeline()` 在 world model 领域的复制——这个抽象的科学贡献有限，价值更多在工程整合和 community alignment
5. **作者贡献列表暴露分工边界**: 大量 contributor 只做 "tests pipelines"，反映这是一次 codebase 整合工作而非深度研究

### 可信评估

#### Artifact 可获取性

- **代码**: inference-only，开源在 [github.com/OpenDCAI/OpenWorldLib](https://github.com/OpenDCAI/OpenWorldLib)；另有三个 extension repo（3D / VLA / simulator）
- **模型权重**: OpenWorldLib 自身不发模型，只 wrap 已有模型的 checkpoint
- **训练细节**: N/A（inference framework，无训练）
- **数据集**: N/A；评测用 LIBERO + AI2-THOR

#### Claim 可验证性

- ✅ "整合了 X / Y / Z 等方法到统一 pipeline"：可通过 github 代码与 docs 验证
- ⚠️ "Hunyuan-WorldPlay achieving the best overall visual performance"：纯 qualitative，无 metric / human eval 数字 / 样本量
- ⚠️ "VGGT 在大 camera 移动下有几何不一致"：定性观察，未给 quantitative geometric error
- ⚠️ World model 定义本身：作为 position 是 reasonable claim，但 "Sora 不算 world simulator" 这种排他判据缺乏可操作的 falsification criterion
- ❌ "OpenWorldLib presents a standardized workflow and evaluation pipeline"：目前看到的 evaluation 完全 qualitative，称不上 "standardized evaluation pipeline"

### Notes

- 这是一篇典型的 **position + tooling** 论文。学术贡献集中在 §1-2 的定义和分类，其余是工程整合
- 作者团队规模（30+ contributors）和分工模式提示这是一次社区式整合，而非小团队深度研究——可以预期后续会持续迭代 codebase 而非追求 paper-level 突破
- 对个人的价值：**作为 world model 子领域 landscape map 用**——通过笔记里点名的 ~20 个方法快速建立对当前 world model 生态的覆盖；不重读
- 立场上 agree 的部分：把 text-to-video 与 world model 切开（Sora 不是 world simulator），这个区分对避免 field 概念膨胀有价值
- 立场上保留的部分：定义把 "long-term memory + interaction + perception" 三件套作为 ground truth，但这三个能力如何共同 work、它们之间的 interface 是什么（这恰恰是 OpenWorldLib 没解决的——Memory 模块还是空模板）
- ❓ 论文反复强调 "next-frame prediction 不是唯一 paradigm"，但全文 §3-4 几乎所有 implicit 生成都还是 next-frame；这个矛盾没解决——如果不是 next-frame，alternative 是什么？

### Rating

**分数**：2 - Frontier
**理由**：作为 world model 方向的 position + tooling 论文，其定义切角（从 core objective 而非 architecture / task）对 "Sora 是不是 world simulator" 类持续争论提供了 reference point，且 codebase 覆盖了当前主流 video gen / 3D / VLA / reasoning 集成方向——这些都是 frontier 层面的贡献。但它不是 Foundation 档：定义本身可操作性边界模糊、Memory 模块流于占位、评测完全 qualitative，科学贡献集中在 wrapper layer 与 community alignment，而非奠基性方法或事实上的标准 benchmark。短期内是 world model landscape map 的有用入口，但不会成为方向必引。
