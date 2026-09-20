import numpy as np


class PositionController:


    def __init__(self):

        # 位置比例系数

        self.kp = np.array(
            [
                1.0,
                1.0,
                2.0
            ]
        )


    def compute_velocity(
        self,
        target,
        current
    ):


        error = target-current


        velocity_cmd = self.kp*error


        # 限制最大速度

        velocity_cmd=np.clip(
            velocity_cmd,
            -2,
            2
        )


        return velocity_cmd
        
