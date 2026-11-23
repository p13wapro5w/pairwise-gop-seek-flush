# pairwise-gop-seek-flush

## 领域

游戏、图形与多媒体

## 主题

GOP 精确 seek

## 说明

关键帧定位、解码器重建与 PTS 重排。

初始状态下 `media/gop_seek_flush.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
