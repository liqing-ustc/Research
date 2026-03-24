# GitHub Pages Website Design Spec

## Overview

为 MindFlow Obsidian vault 搭建一个基于 Quartz v4 的静态网站，部署到 GitHub Pages，支持全文搜索、知识图谱、backlinks 和标签筛选，所有笔记公开发布。

## Goals

- 在浏览器中浏览、搜索所有 vault 笔记（Papers/Ideas/Topics/Meetings/Daily 等）
- 保持 vault 的 markdown 文件结构不变
- 每次 push 到 main 分支时自动构建并部署

## Non-Goals

- 选择性发布（所有笔记全部公开）
- 服务端动态功能
- 评论系统

## Architecture

使用 **Quartz v4** 作为静态站点生成器：
- Quartz 的构建基础设施放在 `website/` 子目录，与 vault 内容分离
- 构建时通过 `--directory ../` 参数指向 vault 根目录作为内容来源
- GitHub Actions 自动构建并推送到 `gh-pages` 分支
- GitHub Pages 从 `gh-pages` 分支发布

## Repo Structure Changes

```
MindFlow/
├── website/                    ← 新增：所有网站相关文件
│   ├── quartz/                 ← Quartz 核心代码
│   ├── quartz.config.ts        ← 站点配置
│   ├── quartz.layout.ts        ← 页面布局
│   └── package.json            ← Node.js 依赖
├── .github/
│   └── workflows/
│       └── deploy.yml          ← 新增：自动部署 workflow
├── index.md                    ← 新增：网站首页
├── Papers/                     ← 不变
├── Ideas/                      ← 不变
├── Topics/                     ← 不变
├── Meetings/                   ← 不变
├── Daily/                      ← 不变
├── Templates/                  ← 不变
├── Resources/                  ← 不变
└── Attachments/                ← 不变
```

新增到 `.gitignore`：
```
website/node_modules/
dist/
```

## Quartz Configuration

### Site Info (`quartz.config.ts`)
- `pageTitle`: "MindFlow"
- `baseUrl`: "liqing-ustc.github.io/MindFlow"
- `locale`: "en-US"

### Plugins 启用
| 功能 | 插件/组件 |
|------|----------|
| Wikilinks 解析 | `ObsidianFlavoredMarkdown`（默认） |
| Mermaid 图表 | `ObsidianFlavoredMarkdown` 内置 |
| 全文搜索 | `ContentIndex` + `Search` component |
| 知识图谱 | `Graph` component |
| Backlinks | `Backlinks` component |
| 标签筛选 | `TagList` + Tags page |
| 文件树导航 | `Explorer` component |

### 排除目录
`Templates/` 目录不发布到网站（模板文件无阅读价值）：
在 `quartz.config.ts` 的 `ignorePatterns` 中添加 `"Templates/**"`。

## GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-node@v4
        with:
          node-version: 22

      - name: Install dependencies
        working-directory: website
        run: npm ci

      - name: Build
        working-directory: website
        run: npx quartz build --directory ../ --output ../dist

      - name: Deploy to gh-pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

## Homepage (index.md)

vault 根目录新增 `index.md`，作为网站首页，内容包括：
- MindFlow 简介
- 各笔记类型的入口链接（Papers、Ideas、Topics 等）
- 快速使用指南

## Deployment Steps

1. 在 GitHub repo Settings → Pages 中，将 Source 设为 `gh-pages` 分支
2. 首次 push 后 GitHub Actions 自动构建，约 2-3 分钟后网站上线
3. 后续每次 push 到 main 自动重新部署

## Site URL

`https://liqing-ustc.github.io/MindFlow`
