# Tag 体系

> 本文件定义 vault 中所有笔记使用的 tag 规范，按研究方向组织。

## 选择原则

- 添加新笔记时，从下表中选取 1-3 个 tag，**按相关性从高到低排列**（第一个 tag 应为笔记的核心主题）。
- 优先选择所属 domain 内的 tag；只在笔记真正跨领域时使用「跨领域」section 的 tag。
- 如需新增 tag，要遵循设计原则，并更新 tag 列表。
- **Meta tag 例外**：`Workbench/daily/` 下的日报 / 周报文件可使用 `daily-papers`、`weekly` 等格式标记，这些不计入研究主题 tag 配额，也不在本 taxonomy 中正式列出。

## 设计原则

1. **粒度适中**：每个 tag 覆盖一个可辨识的研究子方向
2. **正交性**：tag 之间尽量不重叠
3. **稳定性**：以方法和任务为核心，不随具体模型/公司/数据集变化
4. **按 domain 归类**：每个 tag 归到其最主要所属的研究方向；跨领域 tag 单列

## Tag 列表

### Multimodal Understanding

| Tag | 说明 |
|:----|:-----|
| `VLM` | Vision-Language Model（通用视觉语言模型、多模态对齐与理解） |
| `video-LLM` | Video-capable LLM（长视频理解、时序建模、video-text alignment、streaming video） |
| `video-understanding` | 视频理解任务（action recognition、temporal localization、video QA、event detection） |

### AI Agent

| Tag | 说明 |
|:----|:-----|
| `agentic-RL` | Agentic RL（LLM/agent 的 RL 训练，含 RLHF/RLVR、tool-use RL、multi-turn agent RL、self-improvement） |
| `computer-use` | Computer-use agent（桌面/OS 级操作、screenshot+action、跨应用任务自动化） |
| `gui-agent` | GUI agent（GUI grounding、screen understanding、element detection、mobile/desktop UI 交互） |
| `web-agent` | Web agent、信息获取、MCP、浏览器自动化 |
| `auto-research` | AI 自动化科研（含 AI scientist、自动论文生成、科学发现） |

### Embodied AI

#### 任务

| Tag | 说明 |
|:----|:-----|
| `manipulation` | 机器人操作（含 grasping、bimanual 等） |
| `navigation` | 导航任务（含 exploration） |
| `mobile-manipulation` | 移动操作，manipulation + navigation 的交叉 |

#### 模型/方法

| Tag | 说明 |
|:----|:-----|
| `VLA` | Vision-Language-Action 模型 |
| `imitation-learning` | 模仿学习（含 behavior cloning、teleoperation） |
| `diffusion-policy` | Diffusion-based action generation |
| `flow-matching` | Flow matching 生成方法（主要用于 action generation） |
| `VLN` | Vision-Language Navigation 模型 |

#### 感知/表示

| Tag | 说明 |
|:----|:-----|
| `3D-representation` | 3D 场景表示/重建（含 3DGS、NeRF、neural implicit） |
| `spatial-reasoning` | 空间推理（spatial QA、几何/方位/距离推理、affordance reasoning；区别于 `spatial-memory` 的存储侧） |
| `spatial-memory` | 空间记忆（含 topological map、language memory） |
| `scene-understanding` | 场景理解（含 open-vocabulary、CLIP、scene graph、grounding、affordance） |
| `semantic-map` | 语义地图表示 |
| `SLAM` | 同步定位与地图构建 |

#### 能力

| Tag | 说明 |
|:----|:-----|
| `cross-embodiment` | 跨机器人形态迁移 |

#### 硬件

| Tag | 说明 |
|:----|:-----|
| `legged` | 足式机器人 |

### 跨领域（Shared）

> 以下 tag 同时服务于多个 domain，单独列出避免重复。

| Tag | 说明 | 主要涉及 domain |
|:----|:-----|:---------------|
| `LLM` | Large Language Model（语言模型基座、预训练/post-training、reasoning、tool use 等通用方法；视觉相关请用 `VLM`） | Multimodal / Agent / Embodied |
| `world-model` | World model（环境动态建模、video prediction、action-conditioned simulation） | Multimodal / Embodied |
| `embodied-reasoning` | Embodied reasoning（具身推理：把 VLM/LLM 推理能力用于物理任务，含 chain-of-thought for action、physical commonsense） | Multimodal / Embodied / Agent |
| `RL` | 通用强化学习方法与理论（不特指 agent 或 embodied 场景；Agentic RL 请用 `agentic-RL`） | Agent / Embodied |
| `task-planning` | 任务规划与分解（含 hierarchical planning、long-horizon、skill library） | Agent / Embodied |
| `instruction-following` | 自然语言指令跟随、人机交互 | Agent / Embodied |

## 更新记录

- **2026-04-13 (audit)** — 全 vault tag 审计：新增 `spatial-reasoning`（Embodied/感知）、`embodied-reasoning`（跨领域）、`LLM`（跨领域，重新引入：2026-03-26 曾因过宽删除，本次按 use-with-discipline 原则重新加入，仅在笔记真正聚焦语言模型基座/推理/post-training 时使用，避免给所有 LLM-based 论文都打）；明确 `daily-papers`/`weekly` 为 meta tag 例外。修正 13 个文件的非规范 tag（含 `agentic-rl` 大小写、`spatial-representation`/`spatial-intelligence` 等 alias、`Atari`/`game-engine` 等过细 tag、`survey`/`skill-design` 等 meta tag）。
- **2026-04-13** — 按三大研究方向（Multimodal Understanding / AI Agent / Embodied AI）重组 taxonomy，跨领域 tag（`world-model`、`RL`、`task-planning`、`instruction-following`）单列；组内 tag 按对研究兴趣（VLA / Spatial Intelligence / World Model / Agentic RL）的相关性从高到低排序。Multimodal Understanding 新增 `video-LLM`、`video-understanding`；AI Agent 新增 `agentic-RL`、`computer-use`、`gui-agent`，原 `RL` tag 窄化为通用 RL。
- **2026-03-26** — 删除 `LLM` tag（过于宽泛），从 3 篇论文笔记中移除。新增 `auto-research` tag。全面校准：所有论文 tags 与 taxonomy 一致。
- **2026-03-25** — 初始版本。从 79 个 tag 整理为 18 个，重新标注全部 26 个文件。
