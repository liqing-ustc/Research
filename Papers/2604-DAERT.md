---
title: Uncovering Linguistic Fragility in Vision-Language-Action Models via Diversity-Aware Red Teaming
authors: [Baoshun Tong, Haoran He, Ling Pan, Yang Liu, Liang Lin]
institutes: [Sun Yat-sen University, HKUST]
date_publish: 2026-04-07
venue: arXiv
tags: [VLA, agentic-RL, instruction-following]
paper: https://arxiv.org/abs/2604.05595
website:
github:
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] Uncovering Linguistic Fragility in Vision-Language-Action Models via Diversity-Aware Red Teaming
> - **核心**: 用 diversity-aware RL 训练一个 VLM attacker，把同一任务的指令改写成既保留语义又能击垮 VLA 的多样化对抗指令
> - **方法**: DAERT = ROVER-style implicit Q + cascaded reward gates（structural/semantic/length），训 Qwen3-VL-4B 作为 attacker
> - **结果**: 把 π0 在 LIBERO 上的平均成功率从 93.33% 砍到 5.85%，并能 zero-shot 迁移到 OpenVLA / 3D-Diffuser Actor / SimplerEnv
> - **Sources**: [paper](https://arxiv.org/abs/2604.05595)
> - **Rating**: 2 - Frontier（方法本身是 ROVER + cascaded reward 的应用而非奠基，但 "diversity 作为 transferability 前提" 的实证 + "no action" diagnostic 是 VLA robustness 前沿的有价值贡献）

**Key Takeaways:**
1. **VLA 对语言的鲁棒性显著差于直觉**：仅改写指令（不动视觉），就能让 π0 / OpenVLA 这类 SOTA VLA 任务成功率掉 80+ 个百分点，说明它们更多在 pattern match 表层语言而非 compositional grounding。
2. **标准 RL red-teaming 必然 mode collapse**：GRPO 即便加 KL 也会收敛到一两个 "万能 prompt"，CLIP 余弦多样性 7.05 vs DAERT 12.23——单纯压低 success rate 没有信息量，diversity 才是 coverage 的代理。
3. **"Diversity 是 transferability 的前提"**：在 SimplerEnv 跨 domain 跨架构测试中，GRPO（59.5%）甚至打不过 prompt-only ERT（69.2%），而 DAERT 达到 82%——过拟合 source proxy 的攻击不迁移。
4. **诊断 target：先验证模型真的"听话"**：作者用 "no action" prompt 测 π0.5 仍有 54.9% 成功率，判定它退化成 vision-only policy 后弃用，只攻 π0。这是研究 linguistic fragility 时一个被普遍忽视的 sanity check。
5. **Reward 的 cascaded gate 设计**：先 structural → semantic (CrossEncoder ≥0.6) → length，违反前置条件就不调用昂贵的 simulator，把 RL 的样本效率拉起来。

**Teaser. 整体框架。** Attacker（Qwen3-VL-4B）拿 canonical instruction + 初始 RGB，输出改写后的指令；改写指令喂给 frozen VLA 在 simulator 里执行；simulator 的成功/失败信号经 cascaded reward gates 反传给 attacker，用 ROVER-style diversity-aware RL 更新。

![](https://arxiv.org/html/2604.05595v1/images/pipeline.png)

---

## Problem Formulation

把 language-conditioned manipulation 建成 MDP $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\mu,\gamma)$，VLA 策略 $\pi(a_t|s_t,l_{\rm task})$ 从指令 $l_{\rm task}$ + 状态 $s_t=(I_t,q_t)$ 映射到动作。

Embodied Red Teaming 目标：找一组 attack 指令 $\{l_{\rm attack}^i\}$，每个都要满足 (1) 物理可达 (2) 在 VLA 技能集内 (3) 语言上自然，目标函数：

$$
\min_{l_{\rm attack}^{i}\in l_{\rm task}^{\rm FEASIBLE}}\sum_{i=1}^{N}\left[\mathbb{1}_{\rm succ}(\pi,l_{\rm attack}^{i})\right]-\lambda\cdot{\rm Div}(\{l_{\rm attack}^{i}\})
$$

最小化 VLA 成功率，同时最大化指令多样性。**这个 formulation 是这篇论文的关键 framing**——传统 red-teaming 只盯成功率，但作者明确把 diversity 写进 objective。

## Method: DAERT

### 4.1 RL Formulation

把 red teaming 写成熵正则 RL：

$$
\max_{\theta}\mathbb{E}_{l_{\rm attack}\sim p_{\theta}}\big[R(\pi,l_{\rm attack})\big]+\lambda\cdot H(p_{\theta})
$$

等价为带 KL-to-uniform 的形式：reward + $-\lambda \log p_\theta$，理论上能防止单一 prompt 占据全部概率质量。但作者指出标准 RL（GRPO）依然 mode collapse——KL 对 reference 的约束并不等于真正的 policy entropy。

### 4.2 Diversity-Aware Actor-Critic（核心）

借鉴 ROVER（Random Policy Valuation；引文 [9, 10]，本文作者 Haoran He、Ling Pan 自己的工作），不训独立 critic，用 implicit token-level Q：

$$
Q_{\theta}(a_{t}|s_{t})=\rho\Big(\log p_{\theta}(a_{t}|s_{t})-\log p_{\theta_{\text{old}}}(a_{t}|s_{t})\Big)
$$

target 是 reward + 词表 V 上**均匀**的 successor value：

$$
\widehat{Q}(a_{t}|s_{t})=\widetilde{r}+\frac{1}{|V|}\sum_{a_{t+1}\in V}Q_{\theta}(a_{t+1}|s_{t+1})
$$

**含义**：用 uniform policy 的 successor 估值代替 policy-greedy estimate，前缀 token 只要"未来存在多条成功 continuation"就能拿高 Q——这就是 breadth-seeking bias。等价于在 token level 做 quality-diversity 探索。

> ❓ 词表 V 是完整 vocab 还是 top-k？均匀求和 over $|V|\sim 10^5$ 在每个 token 上的计算成本如何摊销？正文没说，估计实际是 sample-based 估计或 top-k 截断，这是个未解释的工程细节。

Group Relative：每个 input 采 n=6 个 rewrite，reward 减组均值标准化（与 GRPO 同），最后用 Bellman MSE 损失：

$$
\mathcal{L}=\mathbb{E}\left[\sum_t \|Q_{\theta}(a_t|s_t)-\text{sg}[\widehat{Q}(a_t|s_t)]\|_2^2\right]
$$

### 4.3 Cascaded Reward Design

三道 gate，**严格串行**：失败一道就 short-circuit 给固定惩罚，省下昂贵的 simulator 调用。

1. **Structural gate**：检查换行 / "Rewrite:" 之类 meta-prefix / 非英语字符。违反 → $r_{\text{struct}}=-0.2$。
2. **Semantic fidelity gate**：CrossEncoder φ（用 stsb-roberta-large）算 $l_{\rm task}$ 与 $l_{\rm attack}$ 的语义相似度，要求 $\phi\geq\tau_{\text{sem}}=0.6$。违反 → $r_{\text{sem}}=-\max(0,0.6-\phi)$。
3. **Length gate**：超过 $L_{\max}=50$ words → $r_{\text{len}}=-\max(0, |l|/L_{\max}-1)$。防止"verbosity hacking"——靠超长上下文撑爆 VLA 而非真的搞语言层面的攻击。

通过全部 gate 才进 simulator，得二元 $f(l_{\rm attack})=1-\mathbb{1}_{\rm succ}$。最终 reward 是带级联门的组合：

$$
R(\pi,l_{\rm attack};l_{\rm task})=f(l_{\rm attack})\cdot\prod_k(1-I_k) + \sum_k\Big(I_k\cdot r_k\cdot\prod_{j=1}^{k-1}(1-I_j)\Big)
$$

> ❓ Semantic threshold 0.6 是怎么定的？太低会丢任务语义（变成换任务），太高会让 attacker 没空间。论文未做 ablation。

## Experiments

### Setup

- **Targets**: π0 (frozen, official ckpt), OpenVLA (LIBERO-finetuned ckpt), 后续测 3D-Diffuser Actor / OpenVLA-7B
- **Attacker**: Qwen3-VL-4B-Instruct，VERL 框架训练
- **Benchmarks**: LIBERO（主），CALVIN / SimplerEnv（zero-shot 迁移）
- **Hyper**: 100 步，group=6，bs=8，lr=1e-6，KL=0.01，entropy=0.001，ρ=1.0
- **Baselines**: Original / ERT (Karnik 2024, GPT-4o prompt) / GRPO / GRPO w/o KL
- **评估协议**：每 task 10 个 rewrite × 5 episodes = 50 episodes/task

### Diagnostic: Target Selection

**Table 1. "no action" prompt 下的 success rate（诊断 VLA 是否真的 condition on language）。**

| Method | Spatial | Object | Goal | Long | Average |
| --- | --- | --- | --- | --- | --- |
| π0 | 22.0 | 30.4 | 1.6 | 16.6 | 17.65 |
| π0.5 | 62.8 | 69.0 | 12.6 | 75.2 | 54.90 |

π0.5 在没有任务文本时还能达 54.9% 成功率——它已退化成 vision-only policy，attack instruction 无意义；π0 掉到 17.65%，说明它真的依赖语言，可作为 red-teaming target。**这是个非常 sharp 的 methodological insight**：研究 linguistic fragility 之前要先证明 target 真的 condition on language。

### Main Results on LIBERO

**Table 2. π0 上的 attack success rate（Succ↓）和 diversity（Cos / LLM↑）。**

| Method | Spatial Succ↓ | Object Succ↓ | Goal Succ↓ | Long Succ↓ | Avg Succ↓ | Avg Cos↑ | Avg LLM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| π0 (original) | 96.4 | 92.6 | 97.0 | 87.3 | **93.33** | – | – |
| π0 + ERT | 78.4 | 80.4 | 29.2 | 74.0 | 65.50 | 10.15 | 6.35 |
| π0 + GRPO | 23.8 | 18.8 | 6.6 | 32.6 | 20.45 | 7.05 | 4.58 |
| π0 + GRPO w/o KL | 17.0 | 13.8 | 3.4 | 11.2 | 11.35 | 5.28 | 3.95 |
| π0 + DAERT | **7.4** | **8.8** | **3.0** | **4.2** | **5.85** | **12.23** | **8.48** |

**关键观察**：
- DAERT 在 attack effectiveness 上最强（5.85% vs ERT 65.50%），同时 diversity 最高（12.23 vs GRPO w/o KL 5.28）。
- GRPO w/o KL 比 GRPO 更猛但更 collapse（cos 5.28 vs 7.05）——印证 KL 是 stabilizer 但不解决 multi-modal exploration 问题。
- LLM-as-Judge 与 CLIP cos 强相关，提供了 cross-validation。

### Transfer to OpenVLA (黑盒，proxy=π0)

**Table 3. 把 π0-训出来的 attacker 直接攻 OpenVLA。**

| Method | Spatial | Object | Goal | Long | Average |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.50 |
| OpenVLA + ERT | 42.2 | 31.6 | 23.4 | 31.4 | 32.15 |
| OpenVLA + GRPO | 29.0 | 15.8 | 10.6 | 12.6 | 17.00 |
| OpenVLA + DAERT | **8.4** | **6.8** | **4.0** | **5.8** | **6.25** |

DAERT 把 OpenVLA 从 76.5% 砍到 6.25%。说明 VLA 之间共享 linguistic vulnerability，universal adversarial example 存在。

### Cross-Architecture / Cross-Domain Transfer

**CALVIN（target: 3D-Diffuser Actor，点云架构 vs π0 2D 图像）：**
DAERT 在 attack-vs-diversity 的 Pareto frontier 上取得 ~60% attack（ERT/GRPO ~45%），且 diversity 最高。

![](https://arxiv.org/html/2604.05595v1/images/transfer2_calvin_result.png)

**SimplerEnv（target: OpenVLA-7B，跨 sim 域 + 跨 embodiment）：**
"Pick Coke Can" 任务，25 states × 4 variants = 100 episodes：
- Original: 24.0% attack rate（baseline failure）
- GRPO: 59.5%（**比 ERT 还差**）
- ERT: 69.2%
- DAERT: **82.0%**

**这是全文最有意义的实验**：GRPO 在大 distribution shift 下崩溃（甚至打不过 prompt-only），证明缺少 diversity 约束的 RL 学到的是 source-specific 的 brittle 攻击；DAERT 的 diversity 让它学到 transferable 的 universal pattern。

### Qualitative Analysis

**Table 4. ERT vs DAERT 改写对比。**

| Task | Method | Instruction |
| --- | --- | --- |
| Object | Original | Pick up the milk and place it in the basket. |
| | ERT | Move towards the red milk carton, secure it with your grip, and set it in the basket on the left. |
| | DAERT | Retrieve the milk carton from its current position, orient it correctly for insertion, and gently deposit it into the woven basket without disturbing any other objects on the floor. |
| Spatial | Original | Pick up the black bowl next to the ramekin and place it on the plate. |
| | ERT | Identify the black bowl adjacent to the silver container, grasp it, and rest it on the plate with stripes. |
| | DAERT | Retrieve the dark-colored bowl positioned adjacent to the small metallic ramekin, precisely orient it, and gently set it atop the circular plate with concentric red stripes. |

**核心区别**：ERT 做 lexical substitution（同义词替换），DAERT 引入**额外的 procedural constraints**（"orient it correctly", "without disturbing other objects"）——这些 compositional 约束超出短指令训练分布，VLA 的 grasping primitive 直接散架。这其实暗示：**VLA 的 fragility 根源不在词汇变化，而在 compositional capability 缺失**——靠 imitation learning 训出来的 short-command policy 没真正学到组合泛化。

PCA 可视化（Figure 3）显示 GRPO 形成紧密 cluster（mode collapse），ERT 偏向 high-prior region，只有 DAERT 在 embedding space 上铺开。

![](https://arxiv.org/html/2604.05595v1/images/pca_instruction_diversity_libero_goal_task_id_8.png)

### Ablation: KL Constraint

GRPO w/o KL 攻击更强（11.35 vs 20.45 succ rate）但 diversity 反而更低（cos 5.28 vs 7.05）。**KL 是 stabilizer 但不是 diversity solution**——validate 了引入 ROVER-style diversity objective 的必要性。

---

## 关联工作

### 基于
- [[2406-OpenVLA|OpenVLA]]: target 模型之一，OpenVLA-7B 在 SimplerEnv 上的 zero-shot transfer 测试用它
- [[2410-Pi0|π0]]: 主 target VLA + attacker 训练时的 proxy 模型
- ROVER (He et al. 2025, [9,10]): 方法直接 inspiration——random policy valuation 的 diversity-aware Q-learning，本文作者团队 prior work

### 对比
- ERT (Karnik et al. 2024, "Embodied Red Teaming"): 主 baseline，prompt-based GPT-4o 改写。本文 framing 把它定为 "training-free / 缺少 adaptability" 的代表
- GRPO (Shao et al. 2024, DeepSeekMath): 标准 RL baseline，本文用它说明 mode collapse 问题；w/o KL 变体单独 ablation

### 方法相关
- [[2504-Pi05|π0.5]]: diagnostic 实验里被排除的 target——用作 "vision-dominant policy" 反例，对 instruction 不敏感
- 3D-Diffuser Actor (Ke et al. 2025): CALVIN 上的迁移 target，验证跨架构泛化
- LIBERO / CALVIN / SimplerEnv: 三个 manipulation benchmark
- Qwen3-VL-4B: attacker policy backbone
- VERL: 训练框架
- CrossEncoder (stsb-roberta-large): semantic gate 的相似度模型
- DeepSeek-R1: LLM-as-Judge 评估器
- CLIP (ViT-B/32): semantic diversity 评估的 embedding 模型
- Predictive Red Teaming (Majumdar et al. 2025, [23]): 另一类 heuristic-based VLA red teaming
- AttackVLA (Li et al. 2025, [18]): 视觉 patch attack on VLA，与本文 linguistic attack 互补

### 同期 / 类似工作
- LIBERO-plus / LIBERO-pro ([7, 37]): 同样关注 VLA linguistic robustness 的 benchmark 类工作，本文可视为 attack-side 的方法贡献
- "Red-Teaming VLA via Quality Diversity Prompt Generation" (arXiv:2603.12510): 几乎同期且同方向的工作，作者未引用——值得后续比较

---

## 论文点评

### Strengths

1. **Diagnostic 严谨**：用 "no action" prompt 验证 target 模型是否 condition on language，避免在 vision-dominant policy（如 π0.5）上做无意义的 linguistic red-teaming。这是个被普遍忽视的 sanity check。
2. **Diversity-as-prerequisite-for-transfer 是有价值的 finding**：SimplerEnv 上 GRPO 反而打不过 ERT 这个反直觉结果，把 "diversity 不只是 nice-to-have"立住了——它是 transferability 的必要条件。
3. **Cascaded reward 的工程设计实用**：先便宜的 structural/semantic 检查，再昂贵的 simulator——RL 训练样本效率的重要 trick。
4. **Reward design 防 verbosity hacking**：明确意识到长指令撑爆 context 是 trivial attack，加 length penalty 阻断这条捷径，让攻击聚焦"真的语言理解"。
5. **黑盒迁移评估 setting 公平**：attacker 只在 π0 + LIBERO 上训，迁到 OpenVLA / 3D-Diffuser / SimplerEnv 都不再训——非常 honest 的 generalization test。

### Weaknesses

1. **方法新颖性有限**：DAERT 本质是 ROVER（[9, 10] 是同组作者的 ICML/2509 工作）+ cascaded reward 的应用——把现成的 diversity-aware RL 套到 embodied red-teaming 场景。论文自己也承认 "we draw inspiration from ROVER"。
2. **Implicit Q 在大词表上的 tractability 没说清**：$\frac{1}{|V|}\sum_{a_{t+1}\in V} Q_\theta$ 在 |V|~10^5 的词表上每 token 都求和不现实，论文未交代是否 top-k 截断或 sample 估计——这个工程细节缺失影响复现。
3. **Naturalness 问题被回避了**：DAERT 生成的指令明显更 verbose、更 "translationese"（作者自己承认），距离真实用户的 casual command 较远——这削弱了 "real-world deployment safety" 的 framing。Appendix A 的辩护（"非母语 / 技术手册"）较弱。
4. **Semantic threshold τ=0.6 是个魔法数字**：CrossEncoder 相似度 0.6 是不是真能保证 action-intention preservation？没有 ablation 也没有人工验证子集。如果 0.6 实际允许大幅语义漂移，那 "VLA fragility" 可能部分被夸大。
5. **没有真机验证 + benchmark 集中在 LIBERO**：所有结果都在 simulator，sim-to-real gap 下 attack 是否仍然 transfer 未知。LIBERO 本身已被批评太简单（论文 [37, 7] 自己引用）。
6. **没有探索 defense**：纯 attack paper，没尝试用生成的对抗指令做 robust training（虽然 intro 提到一句"can be utilized to augment training datasets"）——少了 closing-the-loop 的 contribution。
7. **数值 vs 文本的细节不一致**：Table 2 GRPO w/o KL 的 LLM diversity 是 3.95，竟然比 GRPO 的 4.58 还低，但论文没特别讨论这个内在矛盾——更高的 attack success 反而对应更低的 diversity，是 Pareto trade-off 的额外证据。

### 可信评估

#### Artifact 可获取性
- **代码**: 论文写 "code and the generated data will be available"，目前未发布（搜索无 GitHub repo）
- **模型权重**: 未发布的 attacker checkpoint
- **训练细节**: 关键超参齐全（lr/bs/group size/KL/entropy/ρ/τ_sem/L_max/100 steps），但实现细节如 implicit Q 的 successor 求和方式未交代
- **数据集**: LIBERO / CALVIN / SimplerEnv 全开源；生成的对抗指令承诺开源未兑现

#### Claim 可验证性
- ✅ "把 π0 LIBERO avg success 从 93.33% 砍到 5.85%"：数值清晰，与 baseline 对比合理（Table 2）
- ✅ "diversity 是 transferability 的必要条件"：SimplerEnv GRPO 59.5% < ERT 69.2% < DAERT 82.0% 是个反直觉但 well-controlled 的证据（Figure 2(b)）
- ⚠️ "+59.7% in attack success rate"：相对哪个 baseline、在哪个 benchmark 上的数值？正文未明示出处
- ⚠️ "uncovers universal linguistic vulnerabilities"：universal 的强 claim 仅基于 3 个 target VLA + 3 个 sim benchmark，没 real robot
- ⚠️ "semantically equivalent" rewrites：CrossEncoder 0.6 阈值 + 没人工验证，"语义等价"打折扣
- ❌ "scalable approach to stress-testing VLA agents before real-world deployment"：在 LIBERO/CALVIN sim 上的 attack 与 real-world deployment safety 之间的 gap 未做过任何论证

### Notes

- **核心 takeaway 给我的研究**：如果在做 VLA / instruction following，"no action" diagnostic prompt 是个便宜又重要的 sanity check——能区分模型是真的 follow language 还是靠 visual prior 蒙。
- **方法上的复用价值**：cascaded reward gates 的 "便宜检查 → 昂贵 simulator" 的 short-circuit 设计，对所有 simulator-in-the-loop RL 都适用，不限于 red-teaming。
- **批判性思考**：作者把 fragility 归因于 "linguistic"，但 Table 4 的对比指出真正的攻击向量是**procedural compositional constraints**（"orient it correctly", "without disturbing"），这其实是 task complexity 增加而非 linguistic variation 增加。所以这篇论文揭示的可能是 VLA 的 **compositional generalization** 缺陷，而不是它名义上声称的 "linguistic robustness" 缺陷。这个 reframing 更与 [[2604-DoVLMsTrulyReason|DoVLMsTrulyReason]] / [[2604-CoTDegradesSpatial|CoT degrades spatial]] 这类工作的主线对齐。
- **可疑的 universality claim**：所有 target 都共享 CLIP/SigLIP 类 vision encoder + LLM-style language head——所谓 "universal linguistic vulnerability" 可能只是 "shared backbone vulnerability"。需要测一个完全不同 backbone 的 VLA（如 RDT 或 OFT 变体）才能立得住。
- **复现优先级**：低。方法上是 ROVER + 工程化 reward，自己跑 LIBERO red-teaming 的 ROI 不高；但生成的对抗指令数据集如果开源，**用来做 robust VLA training 的数据增强**很有价值——这才是这条线的下一步动作。

### Rating

**分数**：2 - Frontier
**理由**：方法层面不是奠基（作者自承 DAERT = ROVER + cascaded reward 的应用，Weaknesses #1），不构成 Foundation；但它在 VLA linguistic robustness 这条前沿线上有两个值得被引用的经验贡献——(1) "no action" diagnostic 把 "fragility 研究"的前置条件立起来（Strengths #1），(2) SimplerEnv 上 GRPO < ERT < DAERT 这个反直觉结果把 "diversity 是 transferability 前提" 从口号变成证据（Strengths #2）。不到 3 是因为 benchmark 集中在 LIBERO/CALVIN sim + 没 real robot + 没 defense closing-the-loop；高于 1 是因为这些经验 findings 会成为后续 VLA robustness / red-teaming 工作的必引 baseline。
