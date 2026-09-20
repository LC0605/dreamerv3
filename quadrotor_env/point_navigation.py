while True:


    # 获取无人机状态

    position,velocity=drone.get_state()


    # 计算目标速度

    velocity_cmd = controller.compute_velocity(

        target_position,

        np.array(position)

    )


    # 根据速度产生推力

    thrust = velocity_controller.update(

        velocity_cmd,

        velocity

    )


    drone.apply_thrust(thrust)


    p.stepSimulation()
