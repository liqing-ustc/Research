---
title: "BiCoord: A Bimanual Manipulation Benchmark towards Long-Horizon Spatial-Temporal Coordination"
authors: [Xingyu Peng, Chen Gao, Liankai Jin, Annan Li, Si Liu]
institutes: [Beihang University, Zhongguancun Academy, National University of Singapore]
date_publish: 2026-04-07
venue: arXiv 2604.05831
tags: [manipulation, VLA, diffusion-policy]
paper: https://arxiv.org/abs/2604.05831
website: https://buaa-colalab.github.io/BiCoord/
github: https://github.com/buaa-colalab/BiCoord-Bench
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] BiCoord: A Bimanual Manipulation Benchmark towards Long-Horizon Spatial-Temporal Coordination
> - **核心**: 现有 bimanual benchmark（RoboTwin、RLBench2）任务短、双臂耦合弱，不能反映真实双手操作的空间-时间紧耦合；BiCoord 提出 18 个长程紧耦合任务 + 一组量化"协同度"的指标（SMP/MRD/ARD/STI 等）+ 阶段化标注 + 评测 SSR
> - **方法**: 在 RoboTwin 2.0 基础上，用"任务设计 → coding agent 生成 action code → 仿真验证 → 轨迹与阶段标注"流水线生成 100 条/任务的演示数据；定义 STI =  $\int_0^1 SMP_{<1-d}\,dd$ 把空间距离阈值与同时运动比例耦合在一个标量上
> - **结果**: BiCoord 的 STI=42.16% ≈ RoboTwin/RLBench2 的 4×；DP/RDT/Pi0/OpenVLA-OFT 在单任务平均 SR 仅 33–46%，在多任务设定下普遍腰斩（Pi0: 46.4 → 27.2），暴露推理能力差、精对齐弱、长程稳定性不足
> - **Sources**: [paper](https://arxiv.org/abs/2604.05831) | [website](https://buaa-colalab.github.io/BiCoord/) | [github](https://github.com/buaa-colalab/BiCoord-Bench)
> - **Rating**: 2 - Frontier（STI 指标设计和诊断有新意、开源 checkpoints + dataset 完整，是目前 bimanual 长程协同评测的少数参考之一；但 18 个任务规模小、强依赖 RoboTwin 2.0、发布时间尚短未见广泛后续采纳，未到 Foundation 档）

**Key Takeaways:**
1. **Benchmark gap diagnosed quantitatively**: 用 SMP（同时运动比例）和 ARD（平均相对距离）就能把现有 bimanual benchmark 的"假协同"暴露——RLBench2 SMP 高达 97% 但 ARD 高达 115%（双臂离得很远，并行 ≠ 协同）；RoboTwin2.0 ARD 较低但 SMP 仅 26%（空间近但时间不并行）
2. **STI 是有用的标量指标**: 一个数同时刻画"空间近 + 时间并行"，BiCoord 把它从 ~10% 拉到 42%。这种"benchmark 自身可被一个数评分"的设计值得借鉴
3. **VLA 不是万能解药**: Pi0/RDT/OpenVLA-OFT 在长程紧耦合任务上的 SR 仍只有 40-50%，在 multi-task fine-tune 后还会大幅下降——pretrain 的迁移收益在长程协同维度上远未饱和
4. **Stage-wise SSR 比 SR 信息密度更高**: 多个任务 SR=0% 但 SSR > 30%，意味着策略能完成前几个 sub-goal 但卡在后期——单一 SR 会把这种信号抹平

**Teaser. BiCoord overview：数据生成流水线、Cook 任务的分阶段示例（体现 phased coupling / spatial-temporal constraint / predictive coordination）、STI 与 long-horizon 指标对比。**

![](https://arxiv.org/html/2604.05831v1/x1.png)

---

## 1. Motivation：现有 bimanual benchmark 缺什么

作者把现有 bimanual 仿真 benchmark 的不足拆成两个互相独立的维度：

- **Short-horizon**：RoboTwin / RLBench2 的任务能用几个 motion primitive 完成（pick/place 级），平均只有 1-2 个 stage、2-3 个物体。这跟真实场景的"多 sub-goal 链 + 状态/接触切换"差距大。
- **Loosely coordinated**：双臂"在同一场景里各干各的"，时间不同步或者空间相距很远，缺乏人类双手操作中常见的"一只稳定一只操作"、"动态角色互换"等紧耦合行为。

作者把人类双手协同总结为三类 pattern：

- **Phased coupling**：双臂在 cooperative 与 independent 阶段间切换（如抛接：先靠近，再分开）
- **Spatial-temporal constraints**：同一时刻达到同一区域，或保持特定相对位姿
- **Predictive coordination**：一臂预判另一臂的未来动作并提前规划（如左臂为右臂的甩动腾出空间）

> ❓ 三类 pattern 的划分是描述性的，不是严格的分类——同一段轨迹可能同时具有多种特征。但作为 benchmark 的设计目标 checklist 是合理的。

---

## 2. Quantifying Bimanual Coordination

### 2.1 Preliminaries

观测：head + 两只 wrist 的三路 RGB $I_t = [I_t^{head}, I_t^{left}, I_t^{right}]$。
状态采用 end-effector pose + gripper：

$$
S_t = [E_t^{left}, G_t^{left}, E_t^{right}, G_t^{right}]
$$

其中 $E_t = (p_t, q_t)$，$p_t \in \mathbb{R}^3$ 是 3D 坐标，$q_t \in \mathbb{H}$ 是四元数。

策略 $\pi$ 预测未来 $H$ 步动作：

$$
[A_t, \cdots, A_{t+H-1}] = \pi(T, I_t, S_t)
$$

### 2.2 Spatial-Temporal Coordination Metrics

**空间维度（双臂离得近不近）**：用相对距离避免不同 benchmark 单位差异——

**Equation 1. MRD / ARD**

$$
MRD = \min_{1 \leq t \leq L} \frac{\|p_t^{left} - p_t^{right}\|_2}{\|p_1^{left} - p_1^{right}\|_2}
$$

$$
ARD = \frac{1}{L} \sum_{t=1}^{L} \frac{\|p_t^{left} - p_t^{right}\|_2}{\|p_1^{left} - p_1^{right}\|_2}
$$

分母用初始时刻的双臂距离做归一化——这避免了"任务空间尺度"对比较的污染。MRD/ARD 越小，说明双臂需要在更近距离协作（如 threading a needle 远难于 pressing two buttons）。

**时间维度（双臂动不动得同步）**：定义"是否在动"：

$$
m_t = \begin{cases} 1, & p_t \neq p_{t-1} \lor G_t \text{ is close} \\ 0, & \text{else} \end{cases}
$$

然后 SMT/SMP：

$$
SMT = \sum_{t=1}^{L} m_t^{left} \cdot m_t^{right}, \quad SMP = SMT / L
$$

**空间-时间耦合（核心新指标）STI**：把 SMP 在不同距离阈值 $d$ 下积分——

$$
SMP_{<d} = \frac{1}{L} \sum_{t=1}^{L} m_t^{left} \cdot m_t^{right} \cdot \chi\left(\frac{\|p_t^{left} - p_t^{right}\|_2}{\|p_1^{left} - p_1^{right}\|_2} < d\right)
$$

$$
STI = \int_0^1 SMP_{<1-d}\, \mathrm{d}d
$$

**含义**：STI 把"双臂同时运动"和"双臂空间靠近"两个条件耦合到一个标量上。两个条件同时满足越多，STI 越高。这是 BiCoord 与现有 benchmark 拉开 4× 差距的核心维度。

> ❓ STI 假设运动+靠近 = 协同，但有些任务的协同是"一只静止稳定 + 一只操作"——稳定者 $m_t = 0$ 但仍在贡献协同。SMT/SMP 会把这类纯稳定行为漏算。论文的 Cook 任务中"放炒锅 + 倾倒"应该就有这个问题。

### 2.3 现有 benchmark 的对比

**Table 1. Benchmark 间空间-时间与长程指标对比**

| Benchmark      | Tasks | Stage Eval | SMT ↑ | SMP (%) ↑ | MRD (%) ↓ | ARD (%) ↓ | STI (%) ↑ | TL ↑ | SN ↑ | ON ↑ |
| -------------- | ----- | ---------- | ----- | --------- | --------- | --------- | --------- | ---- | ---- | ---- |
| RLBench2       | 13    | ✗          | 179   | 97.10     | 54.57     | 114.93    | 11.20     | 186  | 1.61 | 2.23 |
| RoboTwin 2.0   | 50    | ✗          | 60    | 26.10     | 63.83     | 82.37     | 8.13      | 221  | 1.64 | 2.02 |
| BiCoord (Ours) | 18    | ✓          | 329   | 92.81     | 29.59     | 55.77     | 42.16     | 361  | 4.27 | 3.66 |

读法：
- RLBench2 SMP 接近 100% 看起来"完美并行"，但 ARD 高达 115%（双臂离得比初始还远）→ 没有空间协同
- RoboTwin 2.0 反过来：ARD 较低、SMP 仅 26% → 缺时间协同
- BiCoord 在两个维度都拉满 → STI 跳到 42.16%（约 4× 提升）

**Figure 2. 不同 benchmark 在 STI 平面上的曲线对比（横轴空间阈值，纵轴受限 SMP）**

![](https://arxiv.org/html/2604.05831v1/x2.png)

RoboTwin 2.0 曲线贴近空间轴 $d$（说明 SMP 太低），RLBench2 贴近 SMP 轴（说明空间约束太松）。BiCoord 曲线位于两轴之间，覆盖面积大 → STI 高。

---

## 3. BiCoord 的构建流水线

**Figure 3. BiCoord 任务一览**

![](https://arxiv.org/html/2604.05831v1/x3.png)

### 3.1 Pipeline

**Figure 4. 三阶段构建流程**

![](https://arxiv.org/html/2604.05831v1/x4.png)

基础环境是 RoboTwin 2.0。三个步骤：

1. **Task Design**：基于 RoboTwin-OD 物体库手工设计 18 个任务，每个都强调高耦合 + 多阶段，并支持多种 embodiment 与场景纹理。
2. **Action Code Generation**：把任务细节 + API 列表喂给 coding agent 生成 action code → 在仿真上跑 10 个随机 seed，要求成功率 ≥ 0.6 才放行；不达标进 human-in-the-loop 环节。
3. **Trajectory Generation & Annotation**：合格的 code 跑 100 条成功轨迹，自动得到阶段化标注（time zone / sub-goal / 双臂行为）。

> ❓ Coding agent 用的是哪个模型、prompt 是什么样的、human-in-loop 改动占比多少——正文没披露。这部分对复现关键。

### 3.2 Features

- **Tight spatio-temporal coordination**: SMP=92.81%、MRD/ARD 比之前低 45.78% / 32.29%、STI 4×
- **Long-horizon**: 平均 4.27 阶段（≈ 3×）、轨迹长度提升 63.35%、物体数提升 64.13%
- **Stage-wise annotation/evaluation**: 每条轨迹按 sub-goal 切分；每个 stage 有分数 $s \in [0,1]$，sum 为 1。新指标 SSR（Stage-Wise Success Rate）：

$$
SSR = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} s_{ij} \cdot c_{ij}
$$

$N$ 是 rollout 数，$M$ 是 stage 数，$c_{ij} \in \{0,1\}$ 表示该 stage 是否完成。

**Figure 5. 任务 / 物体 / embodiment 多样性可视化**

![](https://arxiv.org/html/2604.05831v1/x5.png)

---

## 4. Experiments

### 4.1 Setup

- **Baselines**: [[2410-Pi0|Pi0]]、RDT、[[2502-OpenVLA-OFT|OpenVLA-OFT]]、Diffusion Policy（DP）
- **Single-task**: 沿用 RoboTwin 2.0 配置
- **Multi-task**: 从官方权重 fine-tune 50,000 步，total batch size = 32
- **硬件**: A800 40GB
- **评测**: 每任务 100 rollouts，报告 SR / SSR / TL（TL 仅在成功轨迹上算；任何 SR=0 的任务被排除在 average TL 之外）

### 4.2 Single-task 结果

**Table 2. Single-task SR / SSR / TL（部分代表性任务）**

| Task                    | DP SR | DP SSR | RDT SR | RDT SSR | OFT SR | OFT SSR | Pi0 SR | Pi0 SSR |
| ----------------------- | ----- | ------ | ------ | ------- | ------ | ------- | ------ | ------- |
| Balance Roller          | 34.0  | 67.0   | 39.0   | 69.5    | 89.0   | 94.5    | 62.0   | 79.5    |
| Build Tower With Blocks | 0.0   | 1.5    | 0.0    | 0.5     | 0.0    | 7.0     | 0.0    | 1.5     |
| Cook                    | 8.0   | 31.2   | 27.0   | 48.2    | 26.0   | 33.5    | 35.0   | 57.2    |
| Handover Block W/Bowls  | 1.0   | 49.5   | 0.0    | 45.0    | 0.0    | 19.0    | 2.0    | 50.5    |
| Jigsaw                  | 0.0   | 0.8    | 44.0   | 79.8    | 4.0    | 21.0    | 17.0   | 49.8    |
| Sweep Block             | 83.0  | 83.0   | 13.0   | 13.0    | 22.0   | 22.0    | 91.0   | 91.0    |
| **Average**             | 33.1  | 52.1   | 39.5   | 57.3    | 40.5   | 52.3    | 46.4   | 62.6    |

主要发现：

- **VLA 总体优于 DP**：Pi0 平均 SR 46.4 vs DP 33.1。pretrain 在 embodied AI 同样 work，但提升幅度并不夸张（< 15 pp）
- **DP 更高效**：在能完成的任务上 TL 普遍最短，比如 Place Plate And Cup 上 DP 比 Pi0 短 9.75%
- **Poor reasoning**: Pi0 能完成 Divide Block Tower 的"分块"模板，但 block 颜色/顺序变化时无法把 block 和 landmark 通过颜色关联——尽管训练数据已经包含多种颜色组合。这指向 imitation learning 缺推理能力，建议加 RL 或高层 planning。
- **Deficiency in precise alignment**: Handover Block With Bowls 任务（一只碗倒到另一只碗）所有方法 SR 几乎为 0，因为对齐前就开始倾倒；DP 在这类对齐任务上反而比 VLA 更准。
- **Stage-wise 反映长程脆弱性**: Collect Pens 任务，RDT/OpenVLA-OFT 前几个 pen 都 OK，到第 4 个就崩；Pi0 全程稳。Build Tower With Blocks 随 block 数增加 SR 急剧下降——表明长程是普遍弱点。

**Figure 6. Divide Block Tower 失败案例：颜色/顺序变化后无法关联 landmark**

![](https://arxiv.org/html/2604.05831v1/x6.png)

**Figure 7. Handover Block With Bowls 失败案例：未对齐就开始倾倒**

![](https://arxiv.org/html/2604.05831v1/x7.png)

**Figure 8. Stage-wise success rate 曲线（Collect Pens、Build Tower），随 stage 推进 SR 衰减**

![](https://arxiv.org/html/2604.05831v1/x8.png)

### 4.3 Multi-task 结果

**Table 3. Multi-task average（vs single-task）**

| Method      | Single SR | Multi SR | Δ     |
| ----------- | --------- | -------- | ----- |
| RDT         | 39.5      | 16.9     | -22.6 |
| OpenVLA-OFT | 40.5      | 23.1     | -17.4 |
| Pi0         | 46.4      | 27.2     | -19.2 |

multi-task 普遍腰斩，但少数任务反而提升（OpenVLA-OFT 在 Sweep Block 上从 22 → 44），说明任务间 skill 共享并非全负迁移——存在 negative interference 也存在 positive transfer。作者建议研究"如何识别 / 利用任务间共性"。

### 4.4 Cook 任务全轨迹可视化

**Figure 9. Cook 任务上四种策略的完整轨迹对比**

![](https://arxiv.org/html/2604.05831v1/x9.png)

三个观察：

- **Recovery ability**: Pi0 第一次抓 bread 失败后没有死板地继续，而是绕一圈回来重抓——VLA 不是僵硬复刻 trajectory，可评估 task state 并纠错
- **Stability**: RDT/OpenVLA-OFT 把 skillet 拿斜了；DP 能水平持锅。DP 的失败更多源自不会调整抓取姿态，VLA 的失败更多源自抓取不稳——两者互补
- **Efficiency**: 倾倒时 DP 选择小幅靠近，VLA 选择大幅旋转。TL 上 DP 比 VLA 短 12.91%

---

## 关联工作

### 基于
- **RoboTwin 2.0** (arXiv:2506.18088): BiCoord 直接构建在其仿真环境与 OD 资产之上，复用 RoboTwin 的 action API 与数据生成框架
- **RoboTwin** (CVPR): generative digital twin 思路的源头

### 对比 benchmark
- **RLBench2** (CoRL'24 Workshop): bimanual 扩展 RLBench，BiCoord 用 STI 揭示其"双臂离得远"的协同盲区
- **RoboCerebra** (arXiv:2506.06677): long-horizon manipulation benchmark，但 unimanual focus
- **LIBERO**: lifelong learning benchmark，unimanual

### 方法（被评测的 baseline）
- [[2410-Pi0|Pi0]]: VLM + flow matching action head；本文 single-task 平均 SR 最高（46.4%），但仍卡在 long-horizon 与精对齐
- **RDT-1B**: bimanual diffusion foundation model，统一不同机器人的 action 表征
- [[2502-OpenVLA-OFT|OpenVLA-OFT]]: 用 parallel decoder 替换 diffusion head 提速；本文中 Sweep Block 等任务在 multi-task fine-tune 后反而提升
- **Diffusion Policy (DP)**: visuomotor policy via action diffusion；本文中精对齐 / 稳定性反而强于 VLA

### 方法相关（论文引用的 bimanual 专项工作）
- **AnyBimanual**: 从 unimanual 数据抽 action template 合成 bimanual——和 BiCoord 强调的"紧耦合"形成对比
- **DIF (Decoupled Interaction Framework)**: 用参数级通信实现协同
- **KStar Diffuser**: 用 kinematics + 机器人结构知识降低双臂冲突
- **PPI**: 平衡 spatial awareness 与 movement continuity
- **VoxAct-B、ManiGaussian++**: leader-follower 架构

---

## 论文点评

### Strengths

1. **Quantitative diagnosis 优秀**：MRD/ARD/SMP/STI 的设计直接把"benchmark 的 bimanual 程度"这个一直靠定性描述的问题数值化，并且让现有 benchmark 的"伪协同"暴露得很清楚（RLBench2 SMP 97% 但 ARD 115%）。比单纯说"任务太简单"有说服力得多
2. **STI 这个标量有判别力**: 4× 差距不是噪声，能直接放在论文 abstract 里支持 "harder benchmark" 的 claim
3. **Stage-wise SSR 信息密度高**: 多个 SR=0 但 SSR > 30% 的任务，说明策略卡在后期 sub-goal——这种信息单 SR 看不出来，对后续方法迭代很有价值
4. **Baseline 选得诚实**: DP + 三种 VLA（Pi0、RDT、OpenVLA-OFT）覆盖了 2025-2026 的代表性架构，single + multi-task 双设定也避免了 cherry-picking

### Weaknesses

1. **18 个任务规模偏小**: 对比 RoboTwin 2.0 的 50 个任务，BiCoord 的 18 个 task 多样性受限。虽然作者强调"质而非量"，但当 multi-task fine-tune 收益不稳定时，task 数太少会让结论 noisy
2. **Coding agent pipeline 黑盒**: 用什么模型、prompt 设计、human-in-the-loop 改动比例都没披露。这是数据质量的关键变量，影响其他人复现 BiCoord 风格的 benchmark
3. **STI 漏算"稳定型协同"**: 一只静止 + 一只操作的场景（如 stabilize-to-act），静止臂 $m_t=0$ 但实际在贡献协同——SMP 会低估这种 pattern 的协同度。Cook 任务的"持锅"应该就受影响
4. **VLA 失败归因偏 hand-wavy**: "lack of reasoning" / "rigid imitation learning" 等结论缺更细的 ablation 支持。Divide Block Tower 失败到底是 visual grounding 不行还是 reasoning 不行，论文没区分
5. **没有 finetune 时长 / scale ablation**: 50k step 是否充分？数据量是否够？这些会影响"VLA 在长程协同上效果差"这一结论的强度

### 可信评估

#### Artifact 可获取性

- **代码**: 开源（[github](https://github.com/buaa-colalab/BiCoord-Bench)），基于 RoboTwin 2.0 + 自定义任务
- **模型权重**: 开放——四种 baseline 在 single + multi-task 设定下的 checkpoint 都在 [HuggingFace](https://huggingface.co/Oshwiciqwq/BiCoord-checkpoints)
- **训练细节**: 部分披露——multi-task 50k step、batch=32、A800 40GB，但 single-task 配置只指向 RoboTwin 2.0 文档，coding agent 细节未披露
- **数据集**: 开源（[HuggingFace](https://huggingface.co/datasets/GradiusTwinbee/BiCoord)），含 18 个任务 × 100 轨迹 + stage 标注 + RoboTwin-OD 修改后的 objects 包

#### Claim 可验证性

- ✅ "STI 是 RoboTwin/RLBench2 的 4×"：表 1 数值可复算（指标定义清晰，公式给全）
- ✅ "VLA 单任务平均 SR 33-46%"：表 2 数据完整、可复现（checkpoint + dataset 都开源）
- ⚠️ "VLA 缺 reasoning"：基于 Divide Block Tower 单一任务的定性观察，没有 controlled ablation 区分 visual grounding vs reasoning
- ⚠️ "DP 比 VLA 更高效（TL 短）"：仅在成功轨迹上算 TL，且 DP 平均 SR 最低 → TL 比较的样本基础不对等，可能存在 selection bias
- ⚠️ "multi-task fine-tune 后 SR 普遍下降"：50k step 是否充分未做 ablation；可能只是欠拟合

### Notes

- **STI 的可移植性**: 这个指标可以反过来用——给定一个新提出的 benchmark，算 STI 看它是不是"假的紧耦合"。这种 "benchmark of benchmarks" 的价值可能比 BiCoord 任务本身还高。

- **VLA 的 long-horizon 短板信号**: 单任务 SR 在长程任务上 (>4 stage) 普遍 < 30%，多任务进一步腰斩。这跟最近其它 long-horizon 工作（RoboCerebra 等）的发现一致——不是 BiCoord 独有的弱点，而是当前 VLA 的系统性问题。值得记入 mental model：**pretrain 的迁移收益在长程协同维度上远未饱和**，这是潜在的 research opportunity。

- **可借鉴的设计**: stage-wise score $s \in [0,1]$ 加权累加成 SSR，比简单"完成几个 stage / 总 stage 数"更灵活——可以给关键 stage 加权重。如果以后做长程任务的 benchmark，这种 design 值得复用。

- **疑问**: STI 的积分是数值积分还是闭式？论文没说，github README 也没提。如果是离散 sum，分辨率会影响数值大小——跨 benchmark 比较前需要统一。

- **隐含问题**: 双臂"力的交互"完全没体现在指标里。比如 hand-over 任务里两手的力对抗强度、稳定臂的 contact force——这些 vision-based metric 都看不到。下一步如果要继续做更细的 bimanual 评估，应该补 force-aware metric。

### Rating

**分数**：2 - Frontier
**理由**：属于当前 bimanual 长程协同评测的前沿参考——STI/SMP/ARD 这组指标在 Strengths 里被评估为"把一直定性的问题数值化"并能揭示已有 benchmark 4× 差距，Artifact 可获取性完整（checkpoints + dataset 全开源），是 Pi0/RDT/OpenVLA-OFT 这批 VLA 在 bimanual 长程场景下的少数可复现评测。但距离 Foundation 还差关键两点：一是 Weaknesses 指出 18 个任务规模小、强依赖 RoboTwin 2.0，未能成为独立的 de facto 标准；二是 2026-04 发布时间过近，尚无充足外部信号显示已被主流后续工作采纳，因此不到 3；也不是 Archived，因为它并未被更通用的 benchmark 取代。
