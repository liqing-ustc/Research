# Tag 体系

> 本文件定义 vault 中所有笔记使用的 tag 规范。添加新笔记时，从下表中选取 2-4 个 tag。
> 如需新增 tag，请先更新本文件。

## 设计原则

1. **粒度适中**：每个 tag 覆盖一个可辨识的研究子方向
2. **正交性**：tag 之间尽量不重叠
3. **稳定性**：以方法和任务为核心，不随具体模型/公司/数据集变化

## Tag 列表

### 任务

| Tag | 说明 |
|:----|:-----|
| `manipulation` | 机器人操作（含 grasping、bimanual 等） |
| `navigation` | 导航任务（含 exploration） |
| `mobile-manipulation` | 移动操作，manipulation + navigation 的交叉 |
| `instruction-following` | 自然语言指令跟随、人机交互 |

### 模型/方法

| Tag | 说明 |
|:----|:-----|
| `VLA` | Vision-Language-Action 模型 |
| `VLN` | Vision-Language Navigation 模型 |
| `VLM` | Vision-Language Model（通用视觉语言模型） |
| `LLM` | Large Language Model |
| `flow-matching` | Flow matching 生成方法 |
| `diffusion-policy` | Diffusion-based action generation |
| `imitation-learning` | 模仿学习（含 behavior cloning、teleoperation） |
| `RL` | 强化学习（含 self-improvement） |

### 感知/表示

| Tag | 说明 |
|:----|:-----|
| `SLAM` | 同步定位与地图构建 |
| `scene-understanding` | 场景理解（含 open-vocabulary、CLIP、scene graph、grounding、affordance） |
| `semantic-map` | 语义地图表示 |
| `3D-representation` | 3D 场景表示/重建（含 3DGS、NeRF、neural implicit） |
| `spatial-memory` | 空间记忆（含 topological map、language memory） |

### 能力

| Tag | 说明 |
|:----|:-----|
| `cross-embodiment` | 跨机器人形态迁移 |
| `task-planning` | 任务规划与分解（含 hierarchical planning、long-horizon、skill library） |

### 硬件

| Tag | 说明 |
|:----|:-----|
| `legged` | 足式机器人 |

### 应用领域

| Tag | 说明 |
|:----|:-----|
| `web-agent` | Web agent、信息获取、MCP、浏览器自动化 |

## 更新记录

- **2026-03-25** — 初始版本。从 79 个 tag 整理为 18 个，重新标注全部 26 个文件。
