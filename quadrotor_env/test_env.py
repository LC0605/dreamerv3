from quadrotor import QuadrotorEnv
import time

if __name__ == "__main__":
    # human=弹出图形窗口，direct=无界面训练模式
    env = QuadrotorEnv(render_mode="human")
    try:
        obs, _ = env.reset()
        print("观测维度 shape:", obs.shape)
        total_r = 0
        for i in range(300):
            act = env.action_space.sample()
            obs, r, done, trunc, info = env.step(act)
            total_r += r

            if i % 20 == 0:
                print(f"step {i:3d} | 高度 z={obs[2]:.2f} | reward={r:.2f}")
            if done:
                print(f"回合终止，总奖励：{total_r:.2f}")
                break
            time.sleep(0.01)
    finally:
        # 无论程序正常/崩溃，都会执行关闭仿真，避免锁死GUI
        env.close()

