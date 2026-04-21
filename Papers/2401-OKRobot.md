---
title: "OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics"
authors: [Peiqi Liu, Yaswanth Orru, Jay Vakil, Chris Paxton, Nur Muhammad Mahi Shafiullah, Lerrel Pinto]
institutes: [New York University, AI at Meta]
date_publish: 2024-01-22
venue: arXiv
tags: [mobile-manipulation, scene-understanding, manipulation]
paper: https://arxiv.org/abs/2401.12202
website: https://ok-robot.github.io/
github: https://github.com/ok-robot/ok-robot
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics
> - **核心**: 一个 zero-shot、modular 的 open-knowledge 移动操作系统，用 off-the-shelf VLM + grasp 模型在真实家庭做 pick-and-drop，关键 insight 是 "怎么拼" 比 "用什么模型" 更决定成败
> - **方法**: iPhone 扫房间建 VoxelMap（CLIP+OWL-ViT+SAM）→ 语义查询导航点 → AnyGrasp+LangSam 做 language-filtered grasping → 启发式 dropping；状态机串联，无需任何训练
> - **结果**: 10 个真实家庭、171 个 trial，zero-shot pick-and-drop 成功率 58.5%（去 clutter / 解歧义后达 82.4%），相对 NeurIPS'23 OVMM 冠军 33% 提升约 1.8x
> - **Sources**: [paper](https://arxiv.org/abs/2401.12202) | [website](https://ok-robot.github.io/) | [github](https://github.com/ok-robot/ok-robot)
> - **Rating**: 2 - Frontier（OVMM 方向的代表性 modular 系统 + 标志性 ablation 设计，被主要后续工作作为 baseline，但非 de facto 标准）

**Key Takeaways:**
1. **Open-knowledge stack 在真实家庭可行但脆弱**：finely tuned modular 组合超过 end-to-end SOTA 1.8x，但 multiplicative error 让单模块 80% 的成功率压缩成整体 < 60%。
2. **Pre-trained VLM 适合开放词汇导航，pre-trained grasp 模型适合 mobile manipulation**：CLIP/OWL-ViT/AnyGrasp 不需 fine-tune 即可迁移到家庭，瓶颈不在感知模型本身。
3. **"Naturalness" 是大幅波动来源**：clutter / 语言歧义 / 物理可达性 让成功率从 82% 跌到 58%，论文显式拆解出三种 "cleanup level" 来量化这一点——这是其方法以外的最有价值数据点。
4. **失败模式集中在三处**：semantic memory 错检索（9.3%）、grasp pose 不可达（8.0%）、机器人硬件（7.5%）；扁平物体、透明物体、>1kg 重物是已知 categorical failure。
5. **"Live memory + grasp planning + 用户交互" 是显式的 open problem**：作者把 static map、open-loop grasp、无错误恢复列为下一步关键瓶颈。

**Teaser. 系统总览：iPhone 扫描 → VoxelMap 语义记忆 → 导航 + 抓取 + 放置三段式 pipeline。**

![](https://arxiv.org/html/2401.12202v2/x1.png)

---

## 系统组成

OK-Robot 解决的查询形式："Pick up A (from B) and drop it on/in C"，A 是物体、B/C 是真实家庭中的位置。系统跑在 Hello Robot Stretch 上，由三个子系统拼接：开放词汇导航、开放词汇 RGB-D 抓取、放置启发式。

### Open-Vocabulary Object Navigation

**Scanning**：用 iPhone 上的 Record3D app 拍摄房间 RGB-D 视频（< 1 分钟/房间），导出带位姿的 RGB-D 序列。同时记录地面表面、可导航区域和障碍物。

**Detect + embed**：每帧上跑 OWL-ViT（preliminary 实验里优于 Detic），抽 bounding box / CLIP embedding / detector confidence。bbox 再用 SAM 精化为 mask。物体类别 query 集来自 Scannet200 标签（见 Appendix B 的完整 200 类列表）。

**VoxelMap 构建**：将 mask 用深度+位姿反投影到 3D，得到带 CLIP 向量的点云，体素化到 5cm 分辨率。每个 voxel 用 detector-confidence 加权平均其 CLIP embedding。这是 static representation——一旦建好就不会随机器人操作动态更新（被作者列为 limitation）。

**Querying**："A on B" 被实现为 "A near B"：取 query A 的 top-10 voxels，query B 的 top-50 voxels，算 10x50 的 L2 配对距离，挑最短 (A,B) pair 的 A-point。这种近似定位允许地图低分辨率，且对物体小幅移动鲁棒。

**Navigation point selection**：与 "看一眼" 不同，机器人必须停在 arm-length 范围内才能后续操作。论文用三个 score function 加权挑导航点：

$$
s_1(\overrightarrow{x}) = \|\overrightarrow{x}-\overrightarrow{x_o}\|
$$

$$
s_2(\overrightarrow{x}) = 40 - \min(\|\overrightarrow{x}-\overrightarrow{x_o}\|, 40)
$$

$$
s_3(\overrightarrow{x}) = \begin{cases} 1/\|\overrightarrow{x}-\overrightarrow{x}_{obs}\|, & \text{if } \|\overrightarrow{x}-\overrightarrow{x}_{obs}\|_0 \leq 30 \\ 0, & \text{otherwise} \end{cases}
$$

$$
s(\overrightarrow{x}) = s_1(\overrightarrow{x}) + 8 s_2(\overrightarrow{x}) + 8 s_3(\overrightarrow{x})
$$

**符号说明**：$\overrightarrow{x_o}$ 是目标物体位置，$\overrightarrow{x}_{obs}$ 是最近障碍物。$s_1$ 拉近物体，$s_2$ 限制最大距离上限以保持 manipulator 可达，$s_3$ 远离障碍。

**A\* 路径规划**：把 VoxelMap 投到 2D 10cm×10cm 的 occupancy grid，未观测格也视作不可通行，每个占用点周围 20cm 视为不可通行（机器人本体半径 + 转弯半径），用 $s_3$ 作为 heuristic，得到接近 Voronoi path 的轨迹。

### Open-Vocabulary Grasping

![](https://arxiv.org/html/2401.12202v2/x2.png)

**Figure 3.** 抓取流程：(a) 机器人 POV → (b) AnyGrasp 输出全部候选 grasp → (c) LangSam 用 language query 得到目标 mask → (d) 投影到 mask 内的候选 grasp → (e) 最终选定 grasp。

**Grasp perception**：到达导航点后，把头部 RGB-D 相机指向语义记忆给的物体 3D 位置，捕一帧 RGB-D。点云送入 AnyGrasp，输出 collision-free 的 parallel-jaw grasp 集合，每个含 grasp point / width / height / depth / "graspness score"。

**Language filtering**：LangSam 对 RGB 图做 language-conditioned 分割得到目标 mask，把所有 grasp 投影到图像上，保留落在 mask 内的。再用启发式打分：$\mathcal{S} - \theta^4/10$，其中 $\mathcal{S}$ 是 graspness score，$\theta$ 是 grasp normal 与 floor normal 的夹角。**这一项是关键 hack**：偏好水平 grasp，因为水平 grasp 对 hand-eye 标定误差更鲁棒（垂直 grasp 要求精准标定，跨家庭巡回时频繁失效）。

**Grasp execution**：用预抓取轨迹 $\langle\overrightarrow{p}-0.2\overrightarrow{a},\;\overrightarrow{p}-0.08\overrightarrow{a},\;\overrightarrow{p}-0.04\overrightarrow{a},\;\overrightarrow{p}\rangle$，沿 approach vector $\overrightarrow{a}$ 渐进逼近 grasp point $\overrightarrow{p}$；越接近越慢，避免撞翻轻物体。闭合后抬臂、收回、转腕把物体收紧贴近本体，方便后续导航。

### Dropping Heuristic

不像 [[2306-HomeRobot|HomeRobot]] baseline 假设 drop-off 是平面，OK-Robot 的启发式覆盖凹型容器（sink、bin、box、bag）：

1. LangSam 用 drop language query 分割 head camera 点云
2. 对齐坐标：X 轴朝机器人前方、Z 轴对齐地面法线，并平移到机器人 (x,y)=(0,0)、地面 z=0
3. 在分割点云上取 (x,y) 中位数 $x_m, y_m$
4. 计算 drop 高度：$z_{\max} = 0.2 + \max\{z \mid (x,y,z) \in P_a;\; 0 \leq x \leq x_m;\; |y - y_m| < 0.1\}$
5. 把 gripper 移到 (drop point, $z_{\max}$) 后释放

> ❓ 这个启发式没有显式 reasoning clutter，作者说 "在实验中平均表现良好"——但 placing failure 在 Table I 里其实出现了若干次，这个公式对窄口容器（如 small bag）的鲁棒性应低于开放容器。

### Deployment

整套流程是预定义的线性状态机（navigate to object → grasp → navigate to goal → drop），**不实现错误检测或恢复**——这是后面误差累积的根本原因。新家庭：扫描 < 1 分钟，建 VoxelMap < 5 分钟，10 分钟内可执行第一次 pick-and-drop。

实验协议：每个家庭挑 10-20 件场内现成物体，用 GPT-4V 生成 language query 以减少 experimenter bias，先过滤掉 navigation 找不到的物体，剩下的顺序执行 pick-and-drop 不重置。

---

## 实验

### 主结果

10 个真实家庭、171 trials，zero-shot pick-and-drop 平均 58.5%。在 Pittsburgh 和 Fremont 各复现一次，包括杂乱的家用厨房和大公寓。

### 模块消融

![](https://arxiv.org/html/2401.12202v2/x3.png)

**Figure 5.** 在 3 个 lab 环境对比不同 semantic memory 和 grasping 模块。

| 类别 | 候选 | 结论 |
|:---|:---|:---|
| Semantic memory | VoxelMap / CLIP-Fields / USA-Net | VoxelMap 略优，方差最低（更可靠） |
| Grasping | AnyGrasp / Open Graspness / Contact-GraspNet / Top-down heuristic ([[2306-HomeRobot\|HomeRobot]]) | AnyGrasp 显著最佳，比次优的 top-down 高约 50%（相对） |

值得注意的反直觉点：**heuristic-based top-down grasp 居然击败 open-source AnyGrasp baseline 和 Contact-GraspNet**——说明 "general-purpose grasping model" 离实用还有距离，多数开源模型在 mobile 摆头视角下 generalize 得很差（Contact-GraspNet 在固定俯视 tabletop 训练，到家庭场景几乎给不出有意义 grasp）。

### Clutter / Ambiguity / Affordance 影响

作者设计了三轮 cleanup（none / low / high）逐步去 clutter 和歧义物：

| Cleanup | Navigation error | Manipulation error | Drop error |
|:---|:---|:---|:---|
| none | 15% | 25% | ~constant |
| low | 12% | 16% | ~constant |
| high | 4% | 13% | ~constant |

Drop 模块对歧义/clutter 不敏感（其失败由放置点几何决定，不由语言决定），所以三轮基本恒定。

### 失败模式分解

![](https://arxiv.org/html/2401.12202v2/extracted/5440896/figures/failure_modes.png)

**Figure 4.** 三大模块各自的 long-tail failure breakdown。

三大主因：
1. **Semantic memory 取错物体（9.3%）**：query 措辞敏感，"两个看上去类似的物体" 容易混淆，相似 prompt 一个成功一个失败（图 7）——典型 prompt sensitivity。
2. **Grasp pose 不可达（8.0%）**：AnyGrasp 完全不知道 Stretch gripper 的 joint limit，proposed grasp 超出关节范围或离 base 太远；同一个 grasp trajectory 反复用，没做轨迹规划，撞到小障碍即失败。
3. **机器人硬件（7.5%）**：1kg payload 上限挡掉满瓶洗洁精；远离地面（床中央、高架）的物体够不到；RealSense 标定漂移；轮子小，地毯/地板交界打滑。

Categorical failure：扁平物体（巧克力条、书本）——two-finger gripper 从平面拿不起。

---

## 关联工作

### 基于
- **CLIP-Fields**：semantic memory 的 design 直接继承 CLIP-Fields，把 RGB-D + open-vocab detector + CLIP embedding 注入空间表示
- **[[2306-HomeRobot|HomeRobot]] / OVMM benchmark**：top-down grasp baseline 和 OVMM 任务定义都来自这条线
- **USA-Net**：A* 路径规划的 heuristic 和 affordance 思路借鉴

### 对比
- **UniTeam (NeurIPS'23 OVMM 冠军, 33%)**：OK-Robot 主要的 numerical baseline，但跨 benchmark 比较
- **AnyGrasp / Contact-GraspNet / Open Graspness**：grasp 模块 ablation 的对比对象
- **VoxelMap / CLIP-Fields / USA-Net**：navigation 模块 ablation 的对比对象

### 方法相关
- **OWL-ViT**：作为开放词汇 detector，比 Detic 在 preliminary 实验中表现更好
- **SAM (Segment Anything)**：把 OWL-ViT 的 bbox 精化为 mask
- **LangSam**：language-conditioned 分割，用于 grasp filtering 和 drop 点定位
- **AnyGrasp**：grasp generation 的 workhorse，在 GraspNet 1B grasp 数据集上预训练
- **Record3D**：iPhone 端的 RGB-D 扫描工具
- **GPT-4V**：实验协议中用于标准化 language query 生成
- **[[2309-ConceptGraphs|ConceptGraphs]] / SayPlan / GOAT**：作者明确指出这些 open-scene-graph / LLM-planning 方向是补充而非竞争——OK-Robot 解决的是 "怎么 pick-and-place"，不解决 long-horizon planning

---

## 论文点评

### Strengths

1. **System-level honesty**：标题 "What Really Matters" 不是修辞，正文确实 ablation 出 "组件如何拼接 > 用哪个组件"。这种 systems paper 在 robotics 里稀缺——大多数论文只 sell 自己的新模块。
2. **Cleanup levels 的 ablation 设计精巧**：把 "open vocabulary 泛化能力" 拆成 navigation / manipulation / placing 三段，再叠加三个 cleanup level 的 2D 网格——这个 design 比单一聚合数字提供了 5x 的信息密度，应作为 OVMM 评估范式被借鉴。
3. **诚实暴露 multiplicative error**：明确指出 80% × 80% × 80% ≈ 50%，这是 modular 系统的根本天花板，并把 "live memory + grasp planning + recovery" 写成 explicit open problem 而非掩盖。
4. **真实家庭、新物体**：10 个 NY 家庭 + Pittsburgh + Fremont 复现，171 个 trial 的逐物体 raw data 全部列在 Appendix E（Table I），可独立审计——这种 raw log 公开度在 robotics 实证论文里偏高。
5. **Hello Robot Stretch + Record3D 的成本极低**：硬件平民化让方法可被复现，不依赖 7-DoF arm 或定制 setup。

### Weaknesses

1. **没有真正的 closed-loop**：state machine 是线性 chain，无 error detection、无 recovery、无 retry——失败一次就 abort。作者承认但没尝试。
2. **VoxelMap 是 static 的**：scan 后地图不更新，物体被移动 / 新物体出现都需重扫；这把 OK-Robot 限制为 "single session batch task" 而非 "持续家庭助手"。
3. **Drop heuristic 工程味重、缺少正面 ablation**：dropping 那段是手工拟合的几何启发式，论文没对比其他 placing 策略（如 affordance-based、learned placement），只说 "performs well on average"——但 Table I 里 placing failure 真实存在，缺少量化。
4. **Grasp filter 公式 $\mathcal{S} - \theta^4/10$ 缺乏 derivation**：为什么是 4 次方？为什么除 10？纯调参产物，没解释 $\theta$ 与 calibration error 的定量关系。
5. **Query 由 GPT-4V 生成的协议混淆了 "open vocabulary" 的定义**：真用户在自然交互中可能给出更口语、更歧义的 query；用 GPT-4V 标准化 query 实际上是给 OWL-ViT/CLIP 喂友好分布，inflate 了 zero-shot 数字。
6. **无 baseline 端到端对比**：和 RT-1 / [[2307-RT2|RT-2]] / 任何 VLA 都没有 head-to-head——只对比了模块化前作（[[2306-HomeRobot|HomeRobot]] UniTeam 33%）。这让 "modular vs end-to-end 在 OVMM 上谁强" 的核心问题悬而未决。
7. **"58.5%" 实际是过滤后的成功率**：实验协议 "先过滤 navigation 找不到的物体再做 pick-and-drop" 把一部分 navigation failure 排除在分母外，真正的 end-to-end 成功率应低于报告数字。

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only（[github](https://github.com/ok-robot/ok-robot) 提供 navigation / manipulation / robot control 三模块完整推理代码 + 安装文档）
- **模型权重**: 不训练任何模型；依赖 OWL-ViT、CLIP、SAM、AnyGrasp、LangSam 的官方 checkpoint。**AnyGrasp 需要单独申请 license + checkpoint**（README 显式注明），这是复现的唯一硬卡点。
- **训练细节**: 不适用（zero-shot 系统）；只有评估协议描述
- **数据集**: 无训练数据；评估 raw log（171 trial 的 per-object 结果）公开在 Appendix E Table I

#### Claim 可验证性
- ✅ "10 homes, 58.5% success"：Table I 完整列出所有 trial 的 pick-object × place-location × result，可逐条审计
- ✅ "VoxelMap > CLIP-Fields > USA-Net 在 lab 三环境"：Figure 5 给均值+方差，环境数小但数据公开
- ✅ "AnyGrasp >> Open Graspness / Contact-GraspNet / Top-down"：同上，Figure 5
- ⚠️ "1.8x improvement over prior SOTA"：拿 OK-Robot 在 NY home 的 58.5% 对比 NeurIPS'23 OVMM challenge 的 33%——但两者评估环境（真实家庭 vs HomeRobot OVMM benchmark）和物体集不同，1.8x 是跨 benchmark 比较，不是同条件
- ⚠️ "82.4% on cleaner environments"：定义 "cleaner" 的 cleanup level 是作者自定的，subjective；不同 operator 可能得不同数字
- ⚠️ "Pre-trained VLMs are highly effective for open-vocabulary navigation"：这个 claim 取决于 query 由 GPT-4V 生成；换成真实用户 noisy query，"highly effective" 可能站不住
- ⚠️ "Pre-trained grasping models can be directly applied"：但实测里 AnyGrasp 是闭源的，open-source 替代品（Open Graspness、Contact-GraspNet）都给不出可用 grasp——这个 claim 实际只成立于 AnyGrasp 一个模型

### Notes

- **核心 takeaway 对我自己的研究**：这篇是 systems paper 的好范本——它的价值不在新模型，而在拆解 "modular open-knowledge 系统在真实家庭里的真实瓶颈"。它把 "VLM 够用、grasp 够用、组合方式才是瓶颈" 这个判断打成了实证证据，对未来 VLA 评估也有借鉴：单纯比 success rate 不够，应像本文这样拆 navigation / manipulation / placing × cleanup level 二维 grid。
- **对 VLA 方向的启示**：OK-Robot 的 multiplicative error 论点（80%³ ≈ 50%）其实是 end-to-end VLA 的天然 motivation——如果 modular 必然累积误差，那么 closed-loop end-to-end 应该理论上有上限优势。但 OK-Robot 没做这个对比；现在 (2026-04) 已有 [[2410-Pi0|π0]] / [[2307-RT2|RT-2]] 等，应当回头补这个 gap。
- **"GPT-4V 生成 query" 的方法论问题值得记住**：用 LLM 标准化测试 prompt 会系统性 inflate VLM-based 系统的 zero-shot 数字。任何 OVMM / VLA 评估都应警惕这种 protocol-level 的 distribution alignment。
- **可复现性细节**：AnyGrasp license 是隐性壁垒——这是 open-knowledge 系统最讽刺的一点："open-knowledge" 的 grasp 模块其实是闭源 license 的。
- **Open problem (作者自陈)**：dynamic semantic memory、grasp plan 而非 pose、user-in-the-loop 解歧义、failure recovery、更鲁棒硬件——任意一项都是后续工作的 single-paper-worthy 切入点。

### Rating

**分数**：2 - Frontier
**理由**：它是 OVMM 方向里被广泛引用的代表性 modular 系统——Strengths 里点出的 "cleanup level × 模块" 二维 ablation 设计和 "80%×80%×80%≈50% multiplicative error" 论点，已成为后续 open-vocabulary mobile manipulation 工作（包括 end-to-end VLA 对比）必须对照的参照系，外部也持续作为 baseline 被引用（RAL'24 Demo track 收录、github 维护中）。不够 Foundation 在于：方法本身是 "怎么拼现成模块" 的 systems contribution，CLIP-Fields / HomeRobot / AnyGrasp 这些底层组件才是真正的奠基工作；且 Weaknesses 里的 static VoxelMap / 无 closed-loop / 跨 benchmark 1.8x 等局限让它不会成为方向必读第一篇。高于 Archived 则因为它提供的 failure-mode 结构化证据至今仍被引用，尚未被后续工作完全取代。
