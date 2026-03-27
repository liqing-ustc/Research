## Domain

[描述你的研究领域，如 "Embodied AI, focusing on VLA + mobile manipulation"]

## Expertise

_AI 的能力自评会随使用积累自动更新。初始状态为空。_

## Collaboration Preferences

- **autonomy_level**: moderate
- **report_frequency**: weekly
- **human_review_required**: [abandon direction, start long experiments, modify Domain-Map established knowledge]

## Autopilot Rules

- CAN: read papers, update memory, generate reports, discover new papers, explore new directions based on agenda
- CAN: freely maintain Domain-Map (add, modify, promote, deprecate entries; never delete)
- NEED APPROVAL: start experiments >2h, abandon a research direction, exceed daily API budget
- CANNOT: delete existing notes, modify Human-written content, publish externally
- MUST: log all operations to Workbench/logs/, trigger Reporter mode for major discoveries

## Budget

- **daily_token_limit**: 500000
- **per_cycle_limit**: 50000
- **expensive_action_threshold**: 100000
