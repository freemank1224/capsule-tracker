"""卡尔曼滤波,状态 [cx, cy, a, h, vcx, vcy, va, vh],与原版 ByteTrack 一致。

std_weight_position / std_weight_velocity 可配置 (v1.0 §6.2)。
纯 NumPy 实现。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    std_weight_position: float = 1.0 / 20
    std_weight_velocity: float = 1.0 / 160


class KalmanFilterXYAH:
    """xyah(中心x、中心y、宽高比a、高h)状态的常速卡尔曼滤波。"""

    def __init__(self, cfg: KalmanConfig | None = None):
        self.cfg = cfg or KalmanConfig()
        ndim, dtype = 4, np.float64
        self._motion_mat = np.eye(2 * ndim, dtype=dtype)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = 1.0
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=dtype)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """由未关联的检测框 (cx,cy,a,h) 创建新轨迹状态。"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self.cfg.std_weight_position * measurement[3],
            2 * self.cfg.std_weight_position * measurement[3],
            1e-2,
            2 * self.cfg.std_weight_position * measurement[3],
            10 * self.cfg.std_weight_velocity * measurement[3],
            10 * self.cfg.std_weight_velocity * measurement[3],
            1e-5,
            10 * self.cfg.std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """运行预测步 dt=1。"""
        h = mean[3]
        std_pos = [
            self.cfg.std_weight_position * h,
            self.cfg.std_weight_position * h,
            1e-2,
            self.cfg.std_weight_position * h,
        ]
        std_vel = [
            self.cfg.std_weight_velocity * h,
            self.cfg.std_weight_velocity * h,
            1e-5,
            self.cfg.std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """把状态分布投影到测量空间。"""
        h = mean[3]
        std = [
            self.cfg.std_weight_position * h,
            self.cfg.std_weight_position * h,
            1e-1,
            self.cfg.std_weight_position * h,
        ]
        innov_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innov_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """运行校正步。K = P·H^T·S^-1,S=projected_cov(对称)。"""
        projected_mean, projected_cov = self.project(mean, covariance)

        pht_t = (covariance @ self._update_mat.T).T          # (H·P·H^T)^T -> (4, 8)
        kalman_gain = np.linalg.solve(projected_cov, pht_t).T  # (8, 4)
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        # P' = P - K S K^T(S 对称,标准简化式)
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
    """(x1,y1,x2,y2) -> (cx,cy,a,h)。"""
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]
    cx = tlbr[0] + w / 2.0
    cy = tlbr[1] + h / 2.0
    a = w / h if h > 0 else 1.0
    return np.array([cx, cy, a, h])


def xyah_to_tlbr(mean: np.ndarray) -> np.ndarray:
    """状态前4维 (cx,cy,a,h) -> (x1,y1,x2,y2)。"""
    cx, cy, a, h = mean[:4]
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
