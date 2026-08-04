import time
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from roboticstoolbox.models import Panda

from manipulability import compute_manipulability
from adaptive_damping import compute_adaptive_damping
from adaptive_dls import compute_dls_velocity

client = RemoteAPIClient()
sim = client.require('sim')


robot = Panda()

joint_handles = [
    sim.getObject('/Franka/joint'),
    sim.getObject('/Franka/link2_resp/joint'),
    sim.getObject('/Franka/link3_resp/joint'),
    sim.getObject('/Franka/link4_resp/joint'),
    sim.getObject('/Franka/link5_resp/joint'),
    sim.getObject('/Franka/link6_resp/joint'),
    sim.getObject('/Franka/link7_resp/joint')
]


end_effector = sim.getObject('/Franka/connection')


desired_position = np.array([
    0.45,
    0.20,
    0.50
])

dt = 0.02                 
gain = 0.10                

position_tolerance = 0.005    
max_linear_speed = 0.10        
max_joint_speed = 0.50         


while True:


    q = np.array([
        sim.getJointPosition(joint)
        for joint in joint_handles
    ])


    # -----------------------------------------
    # Current End Effector Position
    # -----------------------------------------

    current_position = np.array(
        sim.getObjectPosition(end_effector, -1)
    )


    # -----------------------------------------
    # Position Error
    # -----------------------------------------

    position_error = desired_position - current_position

    error_norm = np.linalg.norm(position_error)


    # -----------------------------------------
    # Desired Cartesian Velocity
    # -----------------------------------------

    linear_velocity = gain * position_error

    speed = np.linalg.norm(linear_velocity)

    if speed > max_linear_speed:
        linear_velocity = (
            linear_velocity / speed
        ) * max_linear_speed


    # No orientation control
    angular_velocity = np.zeros(3)

    end_effector_velocity = np.concatenate(
        (
            linear_velocity,
            angular_velocity
        )
    )


    # -----------------------------------------
    # Jacobian
    # -----------------------------------------

    J = robot.jacob0(q)


    # -----------------------------------------
    # Manipulability
    # -----------------------------------------

    manipulability = compute_manipulability(J)


    # -----------------------------------------
    # Adaptive Damping
    # -----------------------------------------

    lambda_value = compute_adaptive_damping(
        manipulability
    )


    # -----------------------------------------
    # Adaptive DLS
    # -----------------------------------------

    q_dot = compute_dls_velocity(
        J,
        end_effector_velocity,
        lambda_value
    )


    # -----------------------------------------
    # Limit Joint Velocities
    # -----------------------------------------

    q_dot = np.clip(
        q_dot,
        -max_joint_speed,
        max_joint_speed
    )


    # -----------------------------------------
    # Euler Integration
    # -----------------------------------------

    q_new = q + q_dot * dt


    # -----------------------------------------
    # Send Commands to Robot
    # -----------------------------------------

    for joint, angle in zip(joint_handles, q_new):
        sim.setJointTargetPosition(
            joint,
            angle
        )


    # -----------------------------------------
    # Debug Information
    # -----------------------------------------

    print("-------------------------------------------")
    print(f"Error            : {error_norm:.5f} m")
    print(f"Manipulability   : {manipulability:.6f}")
    print(f"Lambda           : {lambda_value:.4f}")
    print(f"Joint Speed Norm : {np.linalg.norm(q_dot):.4f}")


    # -----------------------------------------
    # Stop Condition
    # -----------------------------------------

    if error_norm < position_tolerance:
        print("\nTarget Reached!")
        break


    # -----------------------------------------
    # Controller Frequency
    # -----------------------------------------

    time.sleep(dt)


print("\nController Finished.")
