from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CF2XParameters:
    mass: float = 0.027
    arm: float = 0.0397
    kf: float = 3.16e-10
    km: float = 7.94e-12
    thrust_to_weight: float = 2.25
    gravity: float = 9.81
    motor_time_constant: float = 0.025

    @property
    def hover_rpm(self) -> float:
        return float(np.sqrt(self.mass * self.gravity / (4.0 * self.kf)))

    @property
    def max_rpm(self) -> float:
        return float(np.sqrt(self.thrust_to_weight * self.mass * self.gravity / (4.0 * self.kf)))


class VelocityAttitudeController:
    """Velocity-to-motor controller with geometric attitude feedback."""

    def __init__(self, parameters: CF2XParameters):
        self.p = parameters
        self.kp_velocity = np.array([2.0, 2.0, 4.0])
        self.ki_velocity = np.array([0.15, 0.15, 0.35])
        self.kp_attitude = np.array([0.0035, 0.0035, 0.0012])
        self.kd_attitude = np.array([0.00018, 0.00018, 0.00012])
        self.velocity_integral = np.zeros(3)
        self.target_yaw = 0.0

    def reset(self, yaw: float = 0.0) -> None:
        self.velocity_integral[:] = 0.0
        self.target_yaw = yaw

    @staticmethod
    def _vee(matrix: np.ndarray) -> np.ndarray:
        return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])

    def compute(
        self,
        velocity_command: np.ndarray,
        yaw_rate_command: float,
        rotation: np.ndarray,
        velocity: np.ndarray,
        body_rates: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        velocity_error = velocity_command - velocity
        self.velocity_integral = np.clip(
            self.velocity_integral + velocity_error * dt,
            [-1.0, -1.0, -0.4],
            [1.0, 1.0, 0.4],
        )
        acceleration = self.kp_velocity * velocity_error + self.ki_velocity * self.velocity_integral
        desired_force = self.p.mass * (acceleration + np.array([0.0, 0.0, self.p.gravity]))
        force_norm = max(float(np.linalg.norm(desired_force)), 1e-6)
        desired_z = desired_force / force_norm
        self.target_yaw += float(yaw_rate_command) * dt
        heading = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw), 0.0])
        desired_y = np.cross(desired_z, heading)
        desired_y /= max(float(np.linalg.norm(desired_y)), 1e-6)
        desired_x = np.cross(desired_y, desired_z)
        desired_rotation = np.column_stack((desired_x, desired_y, desired_z))
        attitude_error = 0.5 * self._vee(desired_rotation.T @ rotation - rotation.T @ desired_rotation)
        desired_rates = np.array([0.0, 0.0, yaw_rate_command])
        torque = -self.kp_attitude * attitude_error - self.kd_attitude * (body_rates - desired_rates)
        collective = max(0.0, float(desired_force @ rotation[:, 2]))
        arm = self.p.arm / np.sqrt(2.0)
        yaw_ratio = self.p.km / self.p.kf
        allocation = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-yaw_ratio, yaw_ratio, -yaw_ratio, yaw_ratio],
            ]
        )
        forces = np.linalg.solve(allocation, np.r_[collective, torque])
        max_force = self.p.kf * self.p.max_rpm**2
        forces = np.clip(forces, 0.0, max_force)
        return np.sqrt(forces / self.p.kf)
