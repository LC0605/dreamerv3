import numpy as np


class QuadrotorEnv:


    def __init__(self):


        # 状态

        self.position = None

        self.velocity = None


        # 目标

        self.goal = np.array(
            [5,5,5],
            dtype=np.float32
        )


        self.max_steps = 300



    def reset(self):


        self.position = np.array(
            [0,0,1],
            dtype=np.float32
        )


        self.velocity = np.zeros(
            3,
            dtype=np.float32
        )


        self.steps=0


        obs=self.get_obs()


        return obs



    def get_obs(self):


        obs=np.concatenate(

            [

            self.position,

            self.velocity,

            self.goal

            ]

        )


        return obs.astype(
            np.float32
        )



    def step(self,action):


        """
        action:

        [vx,vy,vz]

        """

        self.velocity=np.array(
            action
        )


        self.position += (
            self.velocity*0.05
        )


        self.steps+=1



        distance=np.linalg.norm(

            self.goal-self.position

        )


        reward=-distance



        done=False


        if distance<0.3:

            reward+=100

            done=True



        if self.steps>self.max_steps:

            done=True



        obs=self.get_obs()



        return obs,reward,done
