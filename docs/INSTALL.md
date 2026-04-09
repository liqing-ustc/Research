# Installation Guide

## Prerequisites

- [Obsidian](https://obsidian.md/)
- CLI agents, such as [Claude Code](https://claude.ai/code)
- [Git](https://git-scm.com/) (for version control)


## Step 1: Clone the Repository

```bash
git clone https://github.com/liqing-ustc/MindFlow.git
cd MindFlow
```

## Step 2: Install Agent Skills

MindFlow ships with a set of research skills in `skills/`. These need to be linked into `.claude/skills/` so Claude Code can discover them, along with two external skill plugins.

### 2a. Vault skills

Symlink every skill under `skills/` into `.claude/skills/`:

```bash
# From vault root
for skill in skills/*/*; do
  [ -d "$skill" ] && ln -sfn "../../$skill" ".claude/skills/$(basename $skill)"
done
```

Verify the links:

```bash
ls -l .claude/skills/
# Should show symlinks for the vault skills under the folder skills/
```

### 2b. External Skills (Recommended)

Install the following skills under this folder:

| Skill                                                        | Purpose                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| [superpowers](https://github.com/obra/superpowers)           | Development workflow skills (brainstorming, TDD, debugging, code review, etc.) |
| [obsidian-skills](https://github.com/kepano/obsidian-skills) | Obsidian vault management skills (markdown, bases, canvas, CLI)                |

## Step 3: Install Obsidian Plugins

The following obsidian plugins are recommended:

| Plugin                                                    | Purpose                                             |
| --------------------------------------------------------- | --------------------------------------------------- |
| [Claudian](https://github.com/YishenTu/claudian)          | CLI agent integration (Claude Code inside Obsidian) |
| [Obsidian-Git](https://github.com/Vinzent03/obsidian-git) | View source control for your vault                  |


## Step 4: Open as Obsidian Vault

1. Open Obsidian
2. Click **"Open folder as vault"**
3. Select the cloned `MindFlow` directory
4. Trust the vault when prompted (to enable community plugins)

Obsidian settings (templates, attachments, link format) are pre-configured in `.obsidian/`.

## Optional: Published Website (Quartz)

MindFlow supports publishing as a static site via [Quartz v4](https://quartz.jzhao.xyz/):

```bash
# Install Quartz (if not already)
npx quartz create

# Build and preview
npx quartz build --serve
```
