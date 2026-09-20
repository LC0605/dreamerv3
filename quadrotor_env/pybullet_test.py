import pybullet as p
import pybullet_data
import time


# 打开仿真窗口
physicsClient = p.connect(
    p.GUI
)


# 加载资源路径
p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)


# 加载地面
p.loadURDF(
    "plane.urdf"
)


# 设置重力
p.setGravity(
    0,
    0,
    -9.8
)


while True:

    p.stepSimulation()

    time.sleep(
        1/240
    )
    
