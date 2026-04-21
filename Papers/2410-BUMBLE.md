---
title: "BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation"
authors: [Rutav Shah, Albert Yu, Yifeng Zhu, Yuke Zhu, Roberto Martín-Martín]
institutes: [UT Austin]
date_publish: 2024-10-08
venue: ICRA 2025
tags: [mobile-manipulation, embodied-reasoning, task-planning]
paper: https://arxiv.org/abs/2410.06237
website: https://robin-lab.cs.utexas.edu/BUMBLE/
github: https://github.com/UT-Austin-RobIn/BUMBLE
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] BUMBLE: Unifying Reasoning and Acting with VLMs for Building-wide Mobile Manipulation
> - **核心**: 用单一 VLM 作为中央 reasoner，统一感知 / 技能 / 记忆，端到端解决 building-scale 的 mobile manipulation——跨楼层、跨房间、长达 12 步、15 分钟的长程任务
> - **方法**: VLM (GPT-4o) + GSAM 开放词汇感知 + 8 个参数化技能（GoToLandmark / UseElevator / Pickup / PushObjOnGround / OpenDoor / ...）+ 双层记忆（短期：当前 trial 的 state-action history；长期：跨 trial 的失败案例）+ Set-of-Mark + CoT 提示
> - **结果**: 70 trials × 3 buildings × 3 tasks，47.1% 成功率；OfflineSkillDataset 上 80.2%（vs COME 72.6% / Inner-Monologue 61.7%）；用户研究 Likert 3.7/5.0（COME 2.6）
> - **Sources**: [paper](https://arxiv.org/abs/2410.06237) | [website](https://robin-lab.cs.utexas.edu/BUMBLE/) | [github](https://github.com/UT-Austin-RobIn/BUMBLE)
> - **Rating**: 2 - Frontier（building-wide MoMa 是有价值的 problem formulation 扩展 + 真机大规模评估；但 47.1% SR、依赖大量人工先验、任务模板有限，属重要参考而非 de facto 基础）

**Key Takeaways:**
1. **Building-wide MoMa as a benchmark**: 把 mobile manipulation 的范围从单个房间扩到整栋楼（多楼层、电梯、走廊、未知房间），并通过 12-step / 15-min 的 long-horizon 任务暴露出现有 VLM-based 方法的 memory & skill library 短板
2. **双层记忆是关键 trick**: 短期记忆维护 trial 内 state-action 链供 VLM 反思失败；长期记忆跨 trial 累积人工标注的失败案例，本质是把 in-context lesson 当作离线"经验池"
3. **SoM + CoT 是必需的接口**: Ablation 显示去掉 CoT 掉 19.5pt，去掉 SoM 掉 31pt——直接让 VLM 描述目标对象再交给 GSAM 分割效果差很多（"diet Dr. Pepper" 类细粒度失败）
4. **能 scale 到更强的 VLM**: 在 Claude/Gemini/GPT 三个系列、共 8 个 checkpoint 上验证，技能参数预测随 VLM 能力上升单调改善——框架本身随基座增益自动受惠

**Teaser. Building-wide MoMa 的任务示例与 BUMBLE 的执行轨迹。**
<video src="https://robin-lab.cs.utexas.edu/BUMBLE/static/videos/BUMBLE_header_small.mp4" controls muted playsinline width="720"></video>

---

## Building-wide Mobile Manipulation：问题定义

作者把"楼栋级 mobile manipulation"作为 first-class 问题提出来：机器人需要

- 跨楼层（坐电梯）、跨房间（穿门）、长走廊导航
- 处理走廊里的随机障碍（椅子、湿滑警示牌、纸箱）→ 不只是绕开，常需要主动 push / open
- 在杂乱场景里识别开放词汇目标（"diet soda"、"green marker"）
- 串联 ≤ 12 个 ground-truth skill、单次执行 15 分钟（不计 VLM 查询时间）

**Figure 1. Building-wide MoMa 的典型场景。**
![](https://arxiv.org/html/2410.06237v1/x1.png)

> ❓ 12-step / 15-min 这个 horizon 在 mobile manipulation 文献里算长，但相比 Habitat / OVMM 这些离线 benchmark 还是手工设定的 3 个任务模板，并没有 task generator。"building-wide" 更像是规模上的扩展，而非任务多样性的扩展。

---

## 方法：BUMBLE 架构

整体架构如 Figure 3 所示：VLM 作为中心 controller，每一步迭代地 (1) 接收 RGBD 观察 + 任务指令 + 长短期记忆，(2) 预测下一个 subtask + skill，(3) 预测 skill 的参数，(4) 执行并把结果写回短期记忆。

**Figure 3. BUMBLE 架构。** Skill library（左上）+ 短/长期记忆（中下/左下）+ RGBD 感知（左上）→ 参数化技能预测（右上）→ 执行 → 写入短期记忆（右下）。

![](https://arxiv.org/html/2410.06237v1/x3.png)

<video src="https://robin-lab.cs.utexas.edu/BUMBLE/static/videos/BUMBLE_method_video.mp4" controls muted playsinline width="720"></video>

### 感知系统：GSAM + RGBD back-projection

不直接把 RGB 喂给 VLM 做 grounding，而是先用 **Grounded-SAM**（GroundingDINO + SAM-HQ）做开放词汇分割，再用深度图反投影出物体点云、计算物体到机器人的距离。理由是：分割模型在像素级精度上比 VLM 强，且 VLM 不会读 depth/point cloud。

> ❓ 这等于把 "VLM 直接看图说话" 退化为 "VLM 在 SoM 标注的 RGB 上做选择"——简化了 VLM 的任务，但也把感知瓶颈外包给 GSAM。后面 ablation 证实"去掉 SoM 让 VLM 自由描述"会差 31pt，说明 GSAM 在 fine-grained 分割上仍是短板。

### Skill Library：8 个参数化技能

技能列表（按粒度从粗到细）：

| Skill | 参数 | 描述 |
|---|---|---|
| `GoToLandmark` | landmark image | 用 topological map（landmark 图为节点） + 2D occupancy map 做跨房间导航 |
| `NavigateNearObj` | object segmentation | 走到可见 object 附近 |
| `MoveBase` | direction (F/B/L/R) | 30 cm 微调，仅在最后 few meters 用 |
| `Pickup` | object segmentation | 左臂抓取 |
| `PushObjOnGround` | object seg, direction | 推开障碍 / 重排 |
| `OpenDoor` | left/right | 用对应胳膊推门 |
| `CallElevator` | button segmentation | 在当前楼层呼叫电梯 |
| `UseElevator` | button segmentation | 进电梯后选目标楼层按钮 |

每个 skill 有：a) 参数到 low-level 控制的映射（用 ROS gmapping/amcl/move_base + tracikpy 做 IK），b) 给 VLM 的描述 prompt，c) 失败检测（depth NaN、IK 无解、路径阻塞）。`GoToLandmark` 依赖人工预先采集的 landmark image-to-pose 映射。

### 双层记忆

**短期记忆**：当前 trial 内每一步存 `(scene image, subtask, skill name, parameter, success/failure)`，让 VLM 在下一次决策时能看到完整执行历史，从失败步恢复（如抓取失败后用 `MoveBase` 重新对位）。

**长期记忆**：跨 trial 持久化的"错题本"。离线收集每个技能 5 个示例 → 与 ground truth 比较 → 只保留预测错的，附上 VLM 自己生成的"为什么错"分析。最终 MoveBase / UseElevator / PushObjOnGround / 技能选择各拿到 1-3 个错题。**评估期间不更新长期记忆。**

> ❓ "长期记忆" 实际上是 1-3 个 in-context few-shot 错例，量级很小、不在线更新。说"long-term memory" 略 over-claim——这更像 prompt-level few-shot prior，而非真正的持续学习。论文末尾 limitations 也承认存储增长会变 intractable，未来要做 retrieval 或参数化压缩。

### VLM 决策：两步分解

每个决策步分两次 VLM 调用：

1. **Subtask + skill selection**：输入 = 任务指令 + skill 描述 + 当前 RGB（带 SoM 标注）+ 物体距离 + 短期记忆 + 长期记忆。CoT 给出"下一个 subtask 是什么、应该用哪个 skill"。
2. **Skill parameter estimation**：输入 = 预测的 subtask + 该技能的失败错题本 + 候选参数的 SoM 可视化（如 MoveBase 的四个方向箭头、PushObj 的候选 GSAM mask + 推后位置）。CoT 给出最终参数选择。

**Set-of-Mark (SoM)**：把候选物体 / 候选方向用数字 ID 在 RGB 上叠加，让 VLM 通过"选 ID"而不是"描述自然语言"来 grounding——避免 VLM 输出 ambiguous 描述后 GSAM 再次失败的双重不确定性。

**Figure 4. 一次执行轨迹的 trace。** 每步显示 SoM 标注的图像、所选 skill (蓝)、所选参数、文字 rationale；红色为失败后 VLM 自适应换 skill 的步骤。

![](https://arxiv.org/html/2410.06237v1/x4.png)

---

## 实验

### 任务设置

3 个 long-horizon 任务，每个 3 种自然语言改写：

- **Retrieve diet soda can**：如 "Could you grab me a drink that is low in calories?"
- **Retrieve colored marker**：如 "I want to color the sky in my drawing. Can you get me a marker?"
- **Rearrange chairs**：如 "Could you make the seating chairs in the reception area more orderly?"

跨 3 栋大学建筑（不同楼层 / 房间布局 / 视觉外观），10 trials/building/task，总计 90+ 小时。机器人随机初始化在不同楼层、走廊放随机障碍（关闭的门 / 椅子 / 湿滑警示牌 / 纸箱）。干扰物 5-25 个（含未见过的 diet soda 品牌、marker 品牌）。

### 主结果

**Table I. Building-wide tasks 上的成功率（%, 10 trials each）。**

| Method | Marker B1 | Marker B2 | Marker B3 | Soda B1 | Soda B2 | Soda B3 | Chairs B1 | Avg |
|---|---|---|---|---|---|---|---|---|
| Inner-Monologue | 10 | – | – | 0 | – | – | 10 | 6.7 |
| COME | 40 | 30 | 40 | 40 | 30 | 30 | 40 | 35.7 |
| **BUMBLE** | **60** | **40** | **50** | **40** | **50** | **40** | **50** | **47.1** |

Baseline 解释：
- **Inner-Monologue (IM)**：只用语言场景描述，无 RGB、无长期记忆 → 6.7%，太差所以只测了 B1
- **COME**：用 RGB（像 BUMBLE）但无长期记忆 → 35.7%

BUMBLE 比 COME 平均高 12.1pt，作者归因为长期记忆里的"错题本"。

### Skill parameter 准确率（OfflineSkillDataset, ~120 张图）

**Table II. 单步 skill parameter 预测成功率（%）。**

| Skill Parameter | BUMBLE | COME | IM | w/o CoT | w/o SoM |
|---|---|---|---|---|---|
| Pickup (5-10 distractors) | 80.0 | 80.0 | 65.0 | 80.0 | 50.0 |
| Pickup (20-25 distractors) | 65.0 | 65.0 | 65.0 | 60.0 | 40.0 |
| PushObjectOnGround | 81.0 | 70.4 | 56.8 | 64.3 | 81.8 |
| CallElevator (Button) | 95.0 | 75.0 | 60.0 | 40.0 | 25.0 |
| **Average** | **80.2** | 72.6 | 61.7 | 60.7 | 49.2 |

**Ablation 关键观察**：
- **去掉 CoT**：80.2 → 60.7（–19.5pt）。`CallElevator` 上从 95 暴跌到 40——细粒度按钮选择需要逐步推理
- **去掉 SoM**：80.2 → 49.2（–31pt）。`Pickup` 在干扰多时受冲击大；让 VLM 自由描述对象再让 GSAM 分割效果远不如直接选 SoM ID。有趣的是 `PushObjectOnGround` 反而略涨（81.0 → 81.8），可能是因为推动方向不依赖细粒度物体描述
- **干扰物影响**：5-10 → 20-25 时 Pickup 从 80 掉到 65；论文里失败分析也指出 distractor 多时 VLM 错误率从 10% 升到 38.9%

### Scaling with VLM capability

**Figure 5. 8 个 VLM checkpoint 上 BUMBLE 的 skill parameter 准确率。**
![](https://arxiv.org/html/2410.06237v1/x5.png)

测试了 Claude (Haiku-3 / Sonnet-3 / Opus-3 / Sonnet-3.5)、Gemini (Flash-1.5 / Pro-1.5)、GPT (4o-mini / 4o) 三个系列。同一系列内随能力上升单调改善——支持作者"框架不被 VLM 锁死"的论点。

### 失败模式分析

**Figure 6. 70 trials 的成功 / 失败分类。**
![](https://arxiv.org/html/2410.06237v1/x6.png)

失败构成：
- **VLM reasoning 错**（多数）：选错对象（尤其是 distractor 多时）、push 时引发碰撞、按错电梯按钮
- **Sensor 错**：10/38 由 depth NaN 或 lidar 失败导致
- **GSAM 分割错**：尤其是细粒度品牌（"diet Dr. Pepper" vs 普通 Dr. Pepper）

**涌现行为**：BUMBLE 学会用 `MoveBase` 在 sensor failure 后重新对位（如 depth NaN 时往电梯按钮 / 椅子方向挪一步再试）。这是短期记忆 + 通用 skill library 的副产物，不是显式编程的。

### 用户研究

**Figure 7. 5-point Likert 用户评分（n=10）。**
![](https://arxiv.org/html/2410.06237v1/x7.png)

在两个**未见过**的任务（"打翻水了，找点东西清理"、"手机进水了，找东西吸湿气"，需进入新的 shower room）上对比 BUMBLE vs COME。

- BUMBLE Likert avg = **3.7**，COME = 2.6
- BUMBLE 比 COME 减少 22pt 不可恢复失败、12pt 可恢复失败
- 但 33% 的 BUMBLE rollout 被评为 sub-optimal——常见问题是"抓了能用的，但不是最佳的"（VLM 倾向 greedy 选最近 / 最显眼的物体）

**Robot execution 视频集锦：不同初始状态。**
<video src="https://robin-lab.cs.utexas.edu/BUMBLE/static/videos/BUMBLE_diff_start_small.mp4" controls muted playsinline width="720"></video>

**不同任务（重排椅子、取 diet soda、取 marker）。**
<video src="https://robin-lab.cs.utexas.edu/BUMBLE/static/videos/BUMBLE_diff_task_small.mp4" controls muted playsinline width="720"></video>

**不同建筑。**
<video src="https://robin-lab.cs.utexas.edu/BUMBLE/static/videos/BUMBLE_diff_buildings_small.mp4" controls muted playsinline width="720"></video>

---

## 关联工作

### 基于
- **Grounded-SAM (GroundingDINO + SAM-HQ)**：开放词汇分割模型，BUMBLE 的感知前端
- **Set-of-Mark prompting (SoM)**：在图像上叠加数字 ID 让 VLM 选择
- **Chain-of-Thought (CoT)**：中间推理步
- **TeleMoMa (Dass et al. 2024)**：作者实验室的 Tiago robot ROS 基础设施

### 对比
- **Inner-Monologue (Huang et al. 2022)**：纯 LLM 用语言场景描述做规划，BUMBLE 把它扩到带 RGB 后仍弱很多
- **COME (Zhi et al. 2024)**：closed-loop VLM-based MoMa，有 RGB 但无长期记忆，是 BUMBLE 最强 baseline
- **OK-Robot (Liu et al. 2024)**, **HomeRobot**, **Stone et al. 2023**：simple MoMa 的 VLM 方法，缺乏 building-scale 的 skill library 和 memory

### 方法相关
- **TAMP / Neuro-symbolic planning**：经典 long-horizon 方法，受限于 closed-set 物体
- **Interactive navigation (Stilman, HRL4IN, CAMP)**：处理"路径上有障碍需要操作"的设定，但只用几何信息；BUMBLE 加入语义推理
- **VLM for tabletop (MOKA, CoPa)**：方法论近亲，但场景小一个量级

---

## 论文点评

### Strengths

1. **Building-wide MoMa 是有价值的扩展**：把 mobile manipulation 推到楼栋尺度（含电梯、跨楼层）暴露了纯 tabletop / 单房间 baseline 在 long-horizon、interactive navigation 上的 limitation
2. **真机大规模评估**：90+ 小时、3 栋楼、70 trials 是 mobile manipulation 论文里少见的工程量；user study 也比纯定量 SR 更能说明问题
3. **Ablation 很到位**：CoT (-19.5pt) 与 SoM (-31pt) 的拆解清晰量化了 prompting 接口的价值；不同 VLM 上的 scaling curve 支持框架的 future-proof claim
4. **失败分析诚实**：明确给出 sensor / GSAM / VLM reasoning 的归因比例，并承认 greedy plan / 多干扰物 / 细粒度识别的局限

### Weaknesses

1. **"长期记忆" 与名字不符**：实际只是 1-3 个 hand-curated few-shot 错例，且 evaluation 期间不更新。这更像精心调过的 prompt prior，而非真正的 lifelong memory
2. **任务覆盖偏窄**：3 个任务模板（marker / soda / chairs）+ 2 个 unseen 任务。说"unseen" 但仍是 retrieve / arrange 套路，没有真正测试 task family 的开放性
3. **依赖大量人工先验**：landmark images 需人工采集并预先 mapping 到 2D 占据图；skill 列表手工设计；长期记忆错题人工标注。可扩展性受限
4. **47.1% 成功率离 deployable 还很远**：作者把它定位为 "stepping stone" 是诚实的，但用户研究里 33% 的 sub-optimal + ~50% 失败率说明离消费级服务机器人有量级差距
5. **VLM 推理时间不计入 15 min**：对实际部署而言这是关键瓶颈，论文回避了

### 可信评估

#### Artifact 可获取性
- **代码**：开源，inference-only（rw_eval.py 主入口）
- **模型权重**：未训练新权重；依赖 GPT-4o API + 公开的 GroundingDINO (`groundingdino_swinb_cogcoor`) + SAM-HQ (`sam_hq_vit_b`)
- **训练细节**：N/A（无训练）。Prompt 模板写在论文 Appendix VII-A
- **数据集**：OfflineSkillDataset（~120 张人工标注图）未在 GitHub README 中明确开源；landmark 图通过 box.com 链接提供（限作者实验楼）

#### Claim 可验证性
- ✅ **47.1% SR / 80.2% offline acc**：70 trials + Table II 数据，可在公开 dataset 上复现
- ✅ **VLM scaling**：Fig 5 跨 8 个 checkpoint 给出曲线
- ⚠️ **22% 用户满意度提升**：n=10 participants，样本量小；trajectory 是事后图片标注而非 live operation；统计显著性未报告
- ⚠️ **"长期记忆" 带来 12.1pt 提升**：归因不严——COME baseline 没有长期记忆，但也可能 BUMBLE 的其他 prompt 差异（e.g., CoT 模板）共同贡献
- ⚠️ **跨建筑泛化**：3 栋都是 UT Austin 的大学建筑，可能 visual / layout 同构度高
- 无明显的 marketing 话术

### Notes

- "Building-wide" 这个 framing 比方法本身更有 contribution——把 horizon 从单房间推到楼栋是一个清晰的 problem formulation 升级，未来 mobile manipulation 论文可能会沿用这个 benchmark 视角
- 短期 + 长期 memory 的组合很自然但实现很轻量，本质是 "in-context history + in-context error library"。值得思考：当 task / building 规模继续扩大，这个 memory 怎么 scale？dynamic retrieval (RAG over 经验) 或者参数化（fine-tune VLM on errors）是论文 limitations 提到的方向
- VLM-based 系统普遍的 greedy 倾向（user study 里 33% sub-optimal）说明 single-step CoT 不够，未来需要 multi-step lookahead / Monte-Carlo rollout 等更结构化的 planning over VLM
- 与端到端 VLA 路线对比：BUMBLE 走的是 hierarchical—— VLM 做 high-level reasoning + 经典 motion planner 做 low-level，避免了端到端 VLA 在 long-horizon 上的累积误差，但代价是技能库的人工设计与失败模式的拼接复杂度

### Rating

**分数**：2 - Frontier
**理由**：Strengths 显示它把 mobile manipulation 推到了 building-scale，并以 70 trials / 3 栋楼的真机规模 + 细致 ablation 论证了 VLM + memory + SoM 框架，是当前 VLM-based MoMa 方向的重要参考与必比 baseline；但 Weaknesses 中"长期记忆其实只是 1-3 条 few-shot 错例"、3 个手工任务模板、47.1% SR 与大量人工先验，使它并未像 OK-Robot / HomeRobot 那样成为方向的 de facto 奠基，达不到 3 - Foundation；相较"一次性参考"，它的 problem formulation（building-wide MoMa）和系统设计仍具明显 frontier 价值，不属于 1 - Archived。
