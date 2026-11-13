"""计数与稳定性判定 (v1.0 §6.5)。

count = 当前 Tracked 轨迹数;stable = 连续 stable_frames 帧计数不变且无新轨迹;
total_unique = 累计唯一 ID 数(运动场景)。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CountResult:
    count: int
    total_unique: int
    stable: bool
    new_ids: list


class Counter:
    def __init__(self, stable_frames: int = 5):
        self.stable_frames = stable_frames
        self.reset()

    def reset(self) -> None:
        self.seen_ids: set = set()
        self._last_count: int = -1
        self._stable_run: int = 0

    def update(self, tracked_ids: list[int]) -> CountResult:
        new_ids = [i for i in tracked_ids if i not in self.seen_ids]
        self.seen_ids.update(tracked_ids)
        count = len(tracked_ids)
        if count == self._last_count and not new_ids:
            self._stable_run += 1
        else:
            self._stable_run = 0
        self._last_count = count
        return CountResult(
            count=count,
            total_unique=len(self.seen_ids),
            stable=self._stable_run >= self.stable_frames,
            new_ids=new_ids,
        )
