---
title: "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use"
authors: [Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, Tat-Seng Chua]
institutes: [National University of Singapore, East China Normal University, Hong Kong Baptist University]
date_publish: 2025-04-04
venue: Workshop on Reasoning and Planning for Large Language Models (R&P-LLM @ ICLR 2025)
tags: [gui-agent, computer-use]
paper: https://arxiv.org/abs/2504.07981
website: https://gui-agent.github.io/grounding-leaderboard/
github: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use
> - **核心**: 一个面向专业高分辨率桌面应用 (CAD/IDE/创意/科学/办公) 的 GUI grounding benchmark, 揭示现有 grounding 模型在高分辨率小目标场景下的崩溃 (best 仅 18.9%)
> - **方法**: ScreenSeekeR——用 GPT-4o 作 planner 提出候选区域, 递归裁剪缩小搜索空间, 再交给 grounder (OS-Atlas-7B) 定位; 训练免费
> - **结果**: ScreenSeekeR 把 OS-Atlas-7B 从 18.9% 拉到 48.1% (+29.2 abs, +254% rel); 简单的 ReGround crop-and-reground 也能到 40.2%
> - **Sources**: [paper](https://arxiv.org/abs/2504.07981) | [website](https://gui-agent.github.io/grounding-leaderboard/) | [github](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
> - **Rating**: 2 - Frontier（已成为专业高分屏 GUI grounding 的 de facto benchmark, 但 benchmark 属性尚未定型到 Foundation 级别, 且 ScreenSeekeR 方法本身已被后续训练-based 方法超越）

**Key Takeaways:**
1. **Benchmark gap exposed**: 1,581 真实高分辨率截图 (>1080p), 23 个专业应用, 5 类行业 + 3 OS, 平均目标只占整图 0.07% 面积 (vs. ScreenSpot 的 2.01%)——以前的 grounding benchmark 在难度上 underclaim 太多。
2. **Resolution kills grounding**: 即便是 SOTA grounding 模型 ([[2410-OSAtlas|OS-Atlas]]-7B 18.9%, UGround 16.5%, AriaUI 11.3%), 在专业高分屏上几乎全军覆没; GPT-4o 直接 grounding 仅 0.8%。
3. **Search-space reduction is the lever**: 简单的 crop-and-reground (ReGround) 就能把 OS-Atlas-7B 翻倍到 40.2%; 这暗示瓶颈不在模型理解力, 而在 token / 分辨率预算。
4. **Planner 不必会 grounding**: GPT-4o 作 planner (自己 grounding 仅 0.8%) 仍能驱动整个 ScreenSeekeR cascade 到 48.1%——"知道目标大致在哪个 panel" 与 "精确点击像素" 是可分离的能力。
5. **Icon vs. text 是稳定难点**: 所有模型上 icon 准确率显著低于 text (OS-Atlas 28.1% text vs. 4.0% icon), 专业软件的 icon 在 web 训练数据中罕见。

**Teaser. ScreenSpot-Pro 的应用分布与任务数量分桶, 横跨 Development / Creative / CAD / Scientific / Office / OS Commons 六大类。**

![](https://arxiv.org/html/2504.07981v1/x1.png)

---

## 1 问题与动机

GUI agent 当前的研究多集中在 web、移动 app、通用桌面操作等"简单"场景, 而**专业桌面软件** (Photoshop / AutoCAD / MATLAB / Blender / IDE) 几乎被忽略。这类软件有三个独特的难点:

1. **高分辨率显示**: 实际工作流常运行在 ≥1080p 甚至 4K, 当前 MLLM 的 visual encoder 在这个尺度下 token 预算不够。
2. **小目标占比**: 高分辨率下, 目标 UI 元素相对面积更小, 模型 grounding 精度急剧下降。
3. **复杂屏幕环境**: 专业用户常并排打开文档、浏览器、辅助工具, 干扰元素多。

**Figure 2. Grounding 准确率随 target bbox 相对面积单调下降——所有模型 (SeeClick / OS-Atlas / UGround / Qwen2-VL) 在 ScreenSpot-v2 上呈一致趋势。**

![](https://arxiv.org/html/2504.07981v1/x4.png)

> 这张图是文章的核心动机证据: 既然 accuracy 随目标变小线性掉, 那么 "把搜索空间缩小、把目标相对放大" 自然成为最直接的对策。这也直接铺垫了 ScreenSeekeR 的设计。

之前的基准 ([[2401-SeeClick|ScreenSpot]]、VisualWebBench) 在两点上 underclaim: (i) 用裁剪后的局部图代替全屏; (ii) 用候选目标的多选题代替开放定位。专业场景的真实复杂度被绕过了。

---

## 2 ScreenSpot-Pro 数据集

### 2.1 覆盖范围

23 款专业应用, 跨 5 大行业 + 3 OS:

- **Development**: VS Code, PyCharm, Android Studio, Quartus, VMware
- **Creative**: Photoshop, Premiere, Illustrator, Blender, FL Studio, Unreal Engine, DaVinci Resolve
- **CAD/Engineering**: AutoCAD, SolidWorks, Inventor, Vivado
- **Scientific**: MATLAB, Origin, Stata, EViews
- **Office**: Word, Excel, PowerPoint
- **OS Commons**: Windows 11, macOS Sonoma, Ubuntu 24.04

### 2.2 采集方法

- 每款应用由**至少 5 年使用经验的专家**录制, 在真实工作流中触发自定义截屏工具 (热键触发, 截图叠在屏上让标注者拖框 + 输入指令)。
- **统一原生分辨率 ≥1080p, 关闭 monitor scaling, 双屏跨屏截图**。
- **每个 instance 至少 2 名 annotator 复审** (任务有效性 + bbox 精度); 模糊指令必须改写到唯一目标。
- 沿用 [[2401-SeeClick|SeeClick]] 的 text/icon 二分; 但收紧标准——**只有完全无文字 hint 时才标 icon**, 这对 AutoCAD / Office 这种 icon+label 混搭场景是关键的去歧义。

### 2.3 统计

- **1,581 instances**, text:icon ≈ 62.6%:37.4%
- **目标平均占图面积 0.07%** (vs. ScreenSpot 2.01%)——这个 ~30× 的差距是 benchmark 难度的核心来源。
- **附带 ScreenSpot-Pro-CN**: GPT-4 翻译 + 双语作者复审的中文指令版本, 评估双语鲁棒性。

---

## 3 方法: 四种 search-space 缩减策略

ScreenSpot-Pro 的 takeaway 直接驱动 method design——既然小目标是瓶颈, 就把图缩小或迭代裁剪。论文给了 4 个 baseline, 难度由低到高:

### 3.1 Iterative Zooming
受 V\* 启发: 全图先 ground 一次, 把图均匀切成 2×2 patches, 选预测落入的那块继续 ground。

### 3.2 Iterative Narrowing
和 Zooming 同范式, 但 patch **以预测点为中心**裁剪, 大小为原图一半。和并发工作 (Nguyen 2024) 思路一致。

### 3.3 ReGround
最简单: 先 ground, 然后**围绕预测点裁出固定大小 (e.g. 1024×1024) 的区域 re-ground**——给 grounder 一个最优 input size 的局部图。

### 3.4 ScreenSeekeR (主菜)

利用 planner (这里用 GPT-4o) 的 GUI 知识做引导式 visual search:

**Algorithm 主循环**:
1. **Position Inference**: GPT-4o 读全图 + instruction, 用 `<element>/<area>/<neighbor>` XML tag 输出可能位置 (按概率排序)。例如 "delete file" → "可能在 Explorer panel 顶部, 靠近 'new file' 按钮"。
2. **Candidate Scoring**: grounder 在每个候选区域 ground, 用 box-dilation 扩张小框, 再用基于中心距的高斯加权打分:

$$
s = \begin{cases} \exp\Big(-\dfrac{(x'-0.5)^2 + (y'-0.5)^2}{2\sigma^2}\Big) & \text{if point inside} \\ 0 & \text{otherwise} \end{cases}
$$

   其中 $(x', y')$ 是 voting box 中心在候选区域内的归一化坐标, $\sigma=0.3$。直觉是越靠近候选区中心的投票框分数越高, 模拟人眼注意力。NMS 抑制重叠候选。

3. **Recursive Search**: 按分数排序裁出 sub-image 递归调用; 当 patch 小到 ≤1280px 时调 grounder 终判, 由 planner 判断 "is_target / target_elsewhere / target_not_found"。命中即返回, 否则回溯下一个候选。

**Figure 4. ScreenSeekeR (底) vs. plain prediction (左上) vs. ReGround (右上). 任务"delete file or folder", ReGround 被背景的 VS Code 文件 tab 误导, ScreenSeekeR 走出可解释的 search trace (先定位 Explorer 窗口 → 再定位顶栏 → 找到目标)。**

![](https://arxiv.org/html/2504.07981v1/x7.png)

---

## 4 实验结果

### 4.1 End-to-end 模型: 普遍崩溃

**Table 3. End-to-end grounding 模型在 ScreenSpot-Pro 上的 text/icon/avg breakdown (节选)。**

| Model                          | Text Avg | Icon Avg | **Overall Avg** |
| ------------------------------ | -------: | -------: | --------------: |
| OS-Atlas-7B                    |     28.1 |      4.0 |        **18.9** |
| UGround (7B)                   |     25.0 |      2.8 |        **16.5** |
| AriaUI (3.9B act.)             |     17.1 |      2.0 |        **11.3** |
| CogAgent (18B)                 |     12.0 |      0.8 |         **7.7** |
| ShowUI (2B)                    |     10.8 |      2.6 |         **7.7** |
| OS-Atlas-4B                    |      5.0 |      1.7 |         **3.7** |
| Qwen2-VL-7B                    |      2.5 |      0.2 |         **1.6** |
| [[2401-SeeClick\|SeeClick]] (7B) |      1.8 |      0.0 |         **1.1** |
| GPT-4o                         |      1.3 |      0.0 |         **0.8** |
| QwenVL-7B                      |      0.1 |      0.0 |         **0.1** |

观察:
- **Specialist (OS-Atlas / UGround / AriaUI) 显著领先 generalist** (Qwen2-VL, GPT-4o), 即便后者参数量更大、视觉理解更强——专门化的 grounding 训练数据是必要的。
- **GPT-4o 直接 grounding 仅 0.8%**——这个数字反复出现, 说明通用视觉推理 ≠ 像素级 grounding 能力。
- **Icon vs. text 巨大鸿沟**: 即便最强的 OS-Atlas-7B, icon 也只有 4.0%。专业软件的 icon 不在 web 训练分布里, 还经常假定用户已熟悉。

### 4.2 Search 策略对比

**Table 4. 不同 search 策略在 OS-Atlas-7B 之上的提升 (节选, 单位: %)**

| Method                       | Dev  | CAD  | Office | Text | Icon |  **Avg** |
| ---------------------------- | ---: | ---: | -----: | ---: | ---: | -------: |
| OS-Atlas-7B (baseline)       | 17.7 | 10.3 |   27.4 | 28.1 |  4.0 |     18.9 |
| Iterative Focusing           | 33.1 | 23.8 |   43.9 | 43.5 | 10.8 |     31.0 |
| Iterative Narrowing          | 34.4 | 20.3 |   40.9 | 43.5 | 13.1 |     31.9 |
| ReGround                     | 37.5 | 33.3 |   59.1 | 55.7 | 15.1 |     40.2 |
| − Recursive Search (ablate)  | 40.8 | 33.3 |   58.7 | 51.8 | 16.2 |     41.9 |
| − Neighbor Inference (abl.)  | 46.8 | 33.3 |   63.0 | 62.4 | 20.4 |     46.4 |
| − Patch Scoring (ablate)     | 48.5 | 34.1 |   61.3 | 63.3 | 20.2 |     46.8 |
| **ScreenSeekeR (full)**      | 49.8 | 37.9 |   64.3 | 64.1 | 22.4 | **48.1** |

**几个值得注意的点**:
- **ReGround 这个最朴素的 baseline 就能从 18.9 → 40.2**。crop 一刀切的简单操作就能解决一半的问题, 说明 grounder 本身能力在合适分辨率下并不弱, 是输入分辨率把它榨干了。
- **ScreenSeekeR 的提升主要来自 recursive search + neighbor inference + scoring 三个组件的叠加**, 单独 ablate 任何一个都掉 1-2 个点。
- **Icon 仍然只有 22.4%**, 即使加了 search——这说明 search-space 缩减是必要不充分条件, icon 的语义鸿沟需要专门的训练数据补足。

**Table 5. ReGround 的 crop size 消融**——OS-Atlas-7B 在 1024×1024 最优 (40.2%), UGround 在 768×768 最优 (28.8%)——和各模型的 visual encoder 训练分辨率一致。

| Crop Size    | 512×512 | 768×768 | 1024×1024 | 1280×1280 |
| ------------ | ------: | ------: | --------: | --------: |
| OS-Atlas-7B  |    25.1 |    34.2 |  **40.2** |      40.1 |
| UGround (7B) |    27.0 |**28.8** |      28.2 |      26.3 |

> 这是个干净的 "input distribution mismatch" 实验: 模型在 native resolution 工作得最好, 离开就掉。任何 grounding 模型部署时都该 benchmark 这个 sweet spot, 而不是默认全图喂进去。

### 4.3 中文指令: 进一步退化

ScreenSpot-Pro-CN 上 SOTA 的 OS-Atlas-7B 仅 16.8% (英文 18.9%), UGround 从 16.5% 掉到 7.7%。说明大部分 grounding 模型的 multilingual generalization 弱, 训练时英文截图占绝对主导。GPT-4o / QwenVL 略升但基数太小 (0.9% / 0.2%) 无意义。

---

## 关联工作

### 基于
- **[[2401-SeeClick|SeeClick]]**: ScreenSpot 的原始 benchmark + text/icon 二分类标签的来源, ScreenSpot-Pro 沿用其评测协议但把场景升级到专业高分屏
- **V\* (Wu & Xie 2023)**: Iterative visual search for high-resolution images, ScreenSeekeR 的 recursive search 思路直接借鉴

### 对比 (作为 baseline)
- **[[2410-OSAtlas|OS-Atlas]]-7B/4B**: 主力 grounder, ScreenSeekeR 的底座
- **UGround (7B)**: 通用视觉 grounding, 1344×1344 max resolution
- **AriaUI (MoE 3.9B active)**: GUI grounding 专家
- **[[2312-CogAgent|CogAgent]] (18B)**: 早期 GUI VLM
- **[[2411-ShowUI|ShowUI]] (2B)**: 轻量级 GUI VLA
- **Qwen2-VL-7B / QwenVL-7B**: 通用 MLLM, 用作非 grounding-tuned 对照
- **GPT-4o**: 既作 baseline 又作 ScreenSeekeR 的 planner——同一个模型在两个角色的对比 (0.8% vs. 48.1%) 是论文最有意思的实验

### 方法相关
- **Iterative Narrowing (Nguyen 2024)**: 并发独立工作, ScreenSeekeR 的 planner-free baseline 之一
- **[[2404-OSWorld|OSWorld]]**: 端到端 agent 评测 (含 planning + execution), ScreenSpot-Pro 显式选择不重叠这条赛道——分工明确, 一个评 grounding, 一个评 full agent
- **[[2408-OmniParser|OmniParser]]**: 另一类高分辨率 GUI 解析路线 (元素 detection + 描述), 与 ScreenSeekeR 形成 search vs. parse 的方法学对比
- **[[2501-UITARS|UI-TARS]] / [[2509-UITARS2|UI-TARS-2]]**: 后续把 ScreenSpot-Pro 作为 grounding 能力的标准评测之一

---

## 论文点评

### Strengths

1. **Benchmark 切中真实痛点**——专业软件的 GUI 自动化是产业落地的最大蓝海之一 (设计师 / 工程师 / 数据分析师), 而 ScreenSpot 等先前 benchmark 在难度上严重低估了这件事。0.07% vs. 2.01% 的目标占比差距一目了然。
2. **数据采集方法论严谨**: 5 年经验专家 + 真实工作流截屏 + 双人复审 + native resolution + 跨屏处理。这种 "真实分布" 的采集成本高, 但 benchmark 价值正比于此。
3. **方法是 takeaway-driven 的**: 先用 Figure 2 证明 "small target → low accuracy" 是普适规律, 再用 ReGround 证明 "search-space 缩减就能大幅 lift", 再升级到 ScreenSeekeR——claim-evidence-method 链条干净。
4. **ScreenSeekeR 的 planner-grounder 解耦有 transferable insight**: planner 自己 grounding 只有 0.8% 仍能驱动 cascade 到 48.1%, 这暗示 GUI agent stack 可以进一步分工——"知道大致在哪" (semantic) 和 "精准定位像素" (geometric) 用不同模型。
5. **Open & reproducible**: HuggingFace dataset + GitHub 评测脚本 + leaderboard 都在; 有 Chinese variant; 已被 [[2501-UITARS|UI-TARS]] / Qwen2.5-VL / Omniparser v2 / [[2410-OSAtlas|OS-Atlas]] 等后续工作广泛引用为标准 benchmark。

### Weaknesses

1. **仅评估 grounding, 不评估 planning/execution**——作者承认这是为了规避商业软件的 license 风险 ([[2404-OSWorld|OSWorld]] 路线被显式排除)。但这意味着 ScreenSpot-Pro 上的 SOTA 不能直接外推到端到端 agent 性能。
2. **ScreenSeekeR 推理成本不报告**: 多轮 GPT-4o planner + 多次 grounder 调用, 实际延迟和 token 成本对比 baseline 的差距应该是数量级的, 这对部署判断很关键, 但表里没有。
3. **Planner 选择没消融**: 整个 search 框架的智能很大程度依赖 GPT-4o 的 GUI 知识, 换成 GPT-4o-mini / Claude / 开源 VLM 性能如何? 这个 ablation 缺位让 method 的 generalization claim 打折。
4. **Icon 准确率天花板未解决**: ScreenSeekeR 把 icon 从 4.0% 拉到 22.4%, 但相对 text 64.1% 仍有 3× gap; 论文承认是数据问题但没给 actionable 解决方案。
5. **1,581 个样本对 26 个 (软件 × OS) 子类目分布偏稀疏**: 部分子类只有 7-15 个 icon 样本, breakdown 数字方差较大 (e.g. CAD AutoCAD icon 仅 7 个), 单个子类 SOTA 比较容易过拟合 noise。
6. **2D-only**: 没考虑 3D viewport (Blender / Unreal / SolidWorks) 中绕轴旋转、视角变化等动态交互——这些场景在 CAD 工作流里恰恰是 grounding 最难的部分, 但被静态截图回避了。

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only (评测脚本 `run_ss_pro.sh`); ScreenSeekeR 实现在仓库中
- **模型权重**: 不适用——不训练新模型, 复用 OS-Atlas-7B / UGround / GPT-4o
- **训练细节**: 不适用——training-free 方法
- **数据集**: 完全开源, [HuggingFace likaixin/ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro), 还有 Voxel51 镜像和 ScreenSpot-v2-variants 衍生集

#### Claim 可验证性
- ✅ **现有模型在 ScreenSpot-Pro 上 best 仅 18.9%**: Table 2/3 详细 breakdown, 数据公开可复现
- ✅ **ScreenSeekeR 把 OS-Atlas-7B 提升到 48.1%**: ablation 表完整, 实现开源
- ✅ **ReGround crop=1024 时 OS-Atlas-7B 最优**: Table 5 给出干净的分辨率扫描
- ✅ **Icon 显著难于 text**: 跨 11 个模型的稳定 pattern, 难推翻
- ⚠️ **"GPT-4o 仅 0.8%"**: 这个数字依赖于 prompt 设计和 box 输出格式 parsing, 不同 prompt 可能差很多, 但 0.8% 这个量级 (不到 1%) 的结论 robust
- ⚠️ **"ScreenSeekeR 是 SOTA"**: 论文发表时是, 但 2025 年 5 月后被 SE-GUI (47.2% with 7B, 训练 3k 样本即达, README 自报) 等模型超越; SOTA 这种 claim 时效性强, 应理解为"发表时的方法 SOTA"
- ⚠️ **"native resolution >1080p"**: 部分截图来自双屏跨屏拼接, 实际 effective resolution 远超 1080p, 但论文没有完整的 resolution distribution 直方图

### Notes

- **核心 insight 抽象出来**: "在小目标场景下, search-space reduction 是 grounding 的最大 lever, 比换更强 grounder 更有效"——这个观察可以推广到任何高分辨率视觉定位任务 (medical imaging, satellite, document QA)。
- **可分离能力的暗示**: "spatial reasoning"(知道大致在哪个 panel) 和 "pixel-level grounding"(精确点击) 是可独立优化的两种能力。GPT-4o 在前者强后者弱, OS-Atlas 反之, 组合得最好。这对 GUI agent 的 system design 是个明确信号——不要追求 monolithic 模型, 应当 explicit decomposition。
- ❓ **ScreenSeekeR 的 latency / token 成本**: planner 调用 + 多轮 grounder + 检查 prompt, 每个样本可能要 10+ GPT-4o calls。在生产环境 (computer-use agent) 里这种延迟能接受吗? 这是后续工作 ([[2501-UITARS|UI-TARS]] / Qwen2.5-VL) 走 "把 grounding 直接训进单模型" 路线的动机之一。
- ❓ **3D viewport 的 grounding**: ScreenSpot-Pro 包含 Blender / Unreal / SolidWorks, 但任务似乎主要落在 2D UI controls (菜单 / 工具栏), 而真正的 CAD/3D 工作流会涉及 3D 视角下的对象定位——这是个尚未被 cover 的真空。
- **后续追踪**: README 提到 SE-GUI (2025-05) 用 3k 样本 + 7B 模型达到 47.2%, 接近 ScreenSeekeR 的 48.1%——说明 "训一个更好的小模型" 比 "在已有模型上叠 search" 性价比更高, 这是个值得验证的结论。

### Rating

**分数**：2 - Frontier
**理由**: 作为 benchmark, ScreenSpot-Pro 已被 [[2501-UITARS|UI-TARS]] / Qwen2.5-VL / OmniParser v2 / [[2410-OSAtlas|OS-Atlas]] 等后续主要 GUI grounding 工作广泛采用, 是专业高分屏场景下的 de facto 评测, 具备 Frontier 的社区采纳度; 但相比 [[2401-SeeClick|SeeClick]]-ScreenSpot 那样真正奠基性的评测地位仍有差距 (场景专业化也意味着不如原始 ScreenSpot 通用), 尚未升格为 Foundation。ScreenSeekeR 方法层面属于 training-free 的聪明 baseline, 已被训练-based 后续工作 (SE-GUI 等) 性能追平, 不是方向主脉络。综合二者取 Frontier。