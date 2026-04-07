# Cursor 使用流程（极简版）

## 1) 用 Cursor 打开仓库

`git clone` 本仓库后，用 Cursor IDE 打开仓库根目录。  
在 Chat 里把这三个路径告诉 Cursor：

- `CLAUDE.md`
- `AGENTS.md`
- `skills/`

告诉它你想用cursor使用本仓库。

迁移完成后，Cursor 会在仓库里创建：

- `.cursor/rules/identity-and-principles.mdc`
- `.cursor/rules/notebook-structure.mdc`
- `.cursor/rules/skill-router.mdc`

同时会把 `CLAUDE.md` 调整为兼容入口（以 `AGENTS.md` 为准）。

## 2) 在 Obsidian 用 Terminal 调 Cursor Agent 读 paper

在 Obsidian 安装 Terminal 插件后，就可以在 vault 内直接打开 terminal，调用 Cursor Agent 做 paper 阅读总结等操作。
