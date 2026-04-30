# Errors

Command failures and integration errors.

---

## [ERR-20260430-003] bash_windows_path_separator

**Logged**: 2026-04-30T16:25:00+08:00
**Priority**: low
**Status**: pending
**Area**: tests

### Summary
在 PowerShell 中用 `rg --files` 获取 Windows 路径后直接传给 `bash -n`，反斜杠被 bash 当作转义符处理。

### Error
```text
/bin/bash: scriptsrun_stylegan2ada_projector_optional.sh: No such file or directory
```

### Context
- 目的：批量检查 `scripts/*.sh` 语法。
- 原因：Windows 路径分隔符 `\` 传入 bash 后不是安全路径。

### Suggested Fix
传给 bash 前把 `\` 替换为 `/`，或在 PowerShell 中逐个显式使用正斜杠路径。

### Metadata
- Reproducible: yes
- Related Files: scripts/*.sh

---

## [ERR-20260430-002] powershell_quote_escape

**Logged**: 2026-04-30T16:09:00+08:00
**Priority**: low
**Status**: pending
**Area**: tests

### Summary
在 PowerShell 中用 `python -c` 组合 f-string 与字典键转义时引号处理失败。

### Error
```text
SyntaxError: unterminated string literal
```

### Context
- 目的：核验 AnimeGANv2 GitHub 权重文件列表和大小。
- 原因：PowerShell、Python 字符串和 f-string 三层转义叠加，命令可读性差且容易出错。

### Suggested Fix
复杂在线核验脚本使用 PowerShell here-string 管道到 `python -`，不要在一行 `python -c` 中堆叠多层转义。

### Metadata
- Reproducible: yes
- Related Files: scripts/run_animegan2_infer.sh

---

## [ERR-20260430-001] powershell_heredoc

**Logged**: 2026-04-30T15:58:00+08:00
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
在 PowerShell 中误用了类 Unix 的 `python - <<'PY'` heredoc 写法，导致 Python 前向检查没有执行。

### Error
```text
Missing file specification after redirection operator.
The '<' operator is reserved for future use.
```

### Context
- 工作目录：`C:\GAN`
- 目的：运行 DCGAN 生成器和判别器前向形状检查。
- 原因：PowerShell 不支持该 heredoc 语法。

### Suggested Fix
在 PowerShell 环境中使用 `python -c "..."` 或 PowerShell here-string 管道到 `python -`。

### Metadata
- Reproducible: yes
- Related Files: src/dcgan/models.py

---
