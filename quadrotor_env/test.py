from env import QuadrotorEnv



env=QuadrotorEnv()


obs=env.reset()


print(obs)



for i in range(10):


    action=[1,1,1]


    obs,reward,done=env.step(action)


    print(
        obs,
        reward
    )
    
