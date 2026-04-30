# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260430-001] correction

**Logged**: 2026-04-30T15:30:00+08:00
**Priority**: medium
**Status**: pending
**Area**: docs

### Summary
GAN 人像项目路线应以真实人像生成为主，动漫化和风格迁移只能作为增强模块。

### Details
用户纠正了路线定位：任务主要是真实人像生成，但可以加入动漫化/风格迁移内容。后续文档和实现不应把项目改成纯 CycleGAN、自拍动漫化或风格迁移作业。

### Suggested Action
项目文档和报告中固定采用“真实人像生成主线 + 动漫化/风格迁移增强模块”的结构。

### Metadata
- Source: user_feedback
- Related Files: 技术路线_最终方案.md, 项目完成进度.md
- Tags: gan, portrait-generation, style-transfer

---

