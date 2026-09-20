import pybullet as p
import pybullet_data
import time

from quadrotor import Quadrotor



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



while True:


    drone.apply_force(
        15
    )


    p.stepSimulation()


    state=drone.get_state()


    print(
        state["position"]
    )


    time.sleep(
        1/240
    )
