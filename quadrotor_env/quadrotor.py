import pybullet as p


class Quadrotor:


    def __init__(self):

        collisionShape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                0.2,
                0.2,
                0.05
            ]
        )


        visualShape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                0.2,
                0.2,
                0.05
            ]
        )


        self.id = p.createMultiBody(

            baseMass=1,

            baseCollisionShapeIndex=collisionShape,

            baseVisualShapeIndex=visualShape,

            basePosition=[
                0,
                0,
                0
            ]
        )


        self.mass = 1



    # 获取状态

    def get_state(self):


        pos,orn = p.getBasePositionAndOrientation(
            self.id
        )


        vel,omega = p.getBaseVelocity(
            self.id
        )


        return pos,vel



    # 施加推力

    def apply_thrust(self, thrust):


        p.applyExternalForce(

            self.id,

            -1,

            [
                0,
                0,
                thrust
            ],

            [
                0,
                0,
                0
            ],

            p.WORLD_FRAME
        )
