---
title: "SpatialToken VLA: Scene Graph as Native VLM Spatial Tokens"
tags: [VLA, VLM, scene-understanding]
status: raw
linked_project:
date_updated: "2026-03-31"
---
## Hypothesis

若将 3D scene graph 的节点编码为 VLM input sequence 中的 spatial tokens（类似 π₀ 的 action expert tokens），而非文本化后作为 language prompt，则 VLM 在 embodied spatial reasoning 任务上的性能将显著优于 text serialization 方案（如 OmniVLN 的 hierarchical text prompting），因为 spatial tokens 保留了 geometric 信息（3D position, extent, spatial relations）且不占用 language context window。

## Motivation

1. **解决的知识空白**：DomainMaps/SpatialRep.md Open Question #3 （Language-grounded spatial querying）和 Active Debate #1 （Explicit spatial representation vs end-to-end）。现有方案要么将 scene graph 文本化输入 VLM（OmniVLN: 减少 61.7% tokens 但仍是 text），要么不使用显式空间表示（π0.5: implicit in VLM）。VLM-native spatial tokens 是第三条路。
2. **成功的影响**：范式级贡献——一种让 VLM "原生理解" 3D 空间的方法。如果 spatial tokens 有效，所有 VLM-based embodied systems 都可以受益（VLA、VLN、embodied QA 等）。
3. **时机成熟**：OmniVLN（March 2026）已证明 hierarchical scene graph 可以大幅提升 VLM navigation reasoning；VLM-MSGraph（2025）证明 scene graph 可服务 robotic assembly；π₀ 的 action expert tokens 证明 VLM 可以处理非 text 的 domain-specific tokens。

## Related Work

- [[Papers/2507-MTU3D]] - Online query-based spatial memory：每个 query 有 CLIP embedding + 3D bounding box，可直接作为 spatial token 的候选表示。但 MTU3D 的 queries 不进入 VLM，而是由独立的 Spatial Reasoning Transformer 处理。
- [[Papers/2603-MEM]] - MEM 的多尺度记忆（video short-term + language long-term）中缺乏 spatial 维度。Spatial tokens 可作为 MEM 的第三种记忆模态。
- [[Papers/2410-Pi0]] - π₀ 的 action expert tokens 证明 VLM backbone 可以处理非 text 的 continuous tokens（flow matching action），为 spatial tokens 提供架构先例。
- OmniVLN (March 2026, 非 vault 内) - 5-layer Dynamic Scene Graph → multi-resolution text prompting，减少 61.7% tokens 但仍是 text serialization。SpatialToken 进一步：将 graph nodes 编码为 continuous tokens 而非 text。

**Novelty**: 3/5 — OmniVLN 已做了 scene graph → token-efficient VLM input（但仍是 text）；VLM-MSGraph 已将 scene graph 用于 robotic assembly。SpatialToken 的核心差异在于 continuous token encoding（非 text）和 nav+manip 双任务服务。

## Approach sketch

1. **Spatial Token Encoder**：将 scene graph 的每个 node 编码为一个或多个 continuous tokens：
   - 输入：node 的 CLIP embedding (512d) + 3D centroid (3d) + bounding box extent (3d) + navigability flag (1d) + graspability score (1d)
   - 编码器：轻量 MLP projector（类似 VLM 的 vision projector），将上述特征映射到 VLM 的 token embedding space
   - 输出：每个 node → 1 个 spatial token（与 VLM 的 text tokens 同维度）

2. **VLM Integration**：spatial tokens 插入 VLM input sequence，位于 vision tokens 和 language tokens 之间：
   ```
   [vision tokens] [spatial token 1: kitchen_sink] [spatial token 2: red_mug] ... [language instruction tokens]
   ```
   VLM 通过 self-attention 同时 attend to vision、spatial、language tokens。

3. **训练策略**：
   - Stage 1：冻结 VLM，只训练 spatial token encoder，loss = spatial QA（"哪个物体离机器人最近？""从 A 到 B 要经过哪些房间？"）
   - Stage 2：解冻 VLM 最后几层，联合 fine-tune on embodied tasks（navigation + manipulation）
   - 数据：Habitat 2.0 场景 + ConceptGraphs/MTU3D 生成的 scene graph + navigation/manipulation trajectories

4. **对比实验设计**：
   - SpatialToken vs. text serialized scene graph（OmniVLN-style）vs. no spatial input（π0.5-style）
   - 评估：HomeRobot OVMM（nav+manip）、GOAT-Bench（nav）、EmbodiedQA（spatial reasoning）

## Expected outcome

- 在 spatial reasoning QA 上准确率高于 text serialization 方案 10%+（因为保留了 geometric 精度）
- Context window 使用量比 text serialization 减少 50%+（每个 node 1 token vs. ~20 text tokens）
- 在 HomeRobot OVMM 上 SR 优于无 spatial input 的 VLA baseline

## Risk

1. **VLM 兼容性**：预训练 VLM 的 attention 可能无法有效处理 out-of-distribution 的 spatial tokens（非 vision 非 text）。缓解：Stage 1 冻结 VLM 时充分训练 projector，使 spatial tokens 落入 VLM 的已有 embedding manifold。
2. **信息瓶颈**：每个 node 仅 1 个 token 可能无法编码足够的 geometric 信息。缓解：允许 multi-token encoding（每个 node 2-4 tokens）。
3. **Scene graph 质量**：spatial tokens 的质量取决于上游 scene graph 的准确性（segmentation、3D localization）。缓解：用 MTU3D 的 online query memory 作为输入，已有 SOTA 级的性能保证。
4. **Engineering 复杂度高**：需要修改 VLM 架构、设计新 projector、构建 spatial QA 训练数据。这是一个重量级工程项目。
