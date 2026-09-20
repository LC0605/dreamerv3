import pybullet as p
import pybullet_data
import time


from quadrotor import Quadrotor
from controller import PID



p.connect(
    p.GUI
)


p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)


p.loadURDF(
    "plane.urdf"
)


p.setGravity(
    0,
    0,
    -9.8
)



drone=Quadrotor()



# 高度控制器

pid=PID(

    kp=20,

    ki=0,

    kd=10

)



target_height=5



dt=1/240



while True:



    pos,vel=drone.get_state()



    z=pos[2]



    # PID计算额外推力

    thrust = (

        drone.mass*9.8

        +

        pid.update(
            target_height,
            z,
            dt
        )

    )



    drone.apply_thrust(
        thrust
    )



    p.stepSimulation()



    print(
        "height:",
        z,
        "thrust:",
        thrust
    )



    time.sleep(dt)
