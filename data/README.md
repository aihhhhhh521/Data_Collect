# data 目录说明

用于放置原始输入数据（例如 `photos.csv000`, `photos.csv001`, ...）。

默认情况下：
- 若未设置 `DATA_ROOT`，脚本会尝试读取 `scripts/data`（即脚本目录下的 `data` 子目录）。
- 生产环境建议显式设置 `DATA_ROOT` 指向真实数据路径。

本目录可按你的实际数据组织方式调整。