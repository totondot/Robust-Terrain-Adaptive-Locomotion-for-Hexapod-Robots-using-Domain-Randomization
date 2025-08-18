import os
import time
import pybullet as p
import pybullet_data
import numpy as np
from hexapod_env.hexapod_env import HexapodEnv

# --- Configuration ---
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hexapod_env", "pexod.urdf")
TEXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hexapod_env", "checkerboard.png")

# Simulation parameters
CONTROL_RATE = 240
TIME_STEP = 1.0 / CONTROL_RATE

# Gait parameters (tune these for different walking styles)
GAIT_PERIOD = 240  # Number of physics steps for one full cycle (1 second at 240Hz)
STEP_HEIGHT = 0.1  # How high the foot lifts off the ground (in meters)
FORWARD_STRIDE = 0.15  # How far the foot swings forward (in meters)
TURNING_STRIDE = 0.15  # How far the foot swings sideways for turning (in meters)

# Phase offsets for a wave gait
# This ensures that only one leg is ever in the air at a time, for maximum stability.
# The order is: Front-Left, Middle-Left, Rear-Left, Rear-Right, Middle-Right, Front-Right
GAIT_PHASES = [0, 2 * GAIT_PERIOD / 6, 4 * GAIT_PERIOD / 6, 3 * GAIT_PERIOD / 6, 5 * GAIT_PERIOD / 6, 1 * GAIT_PERIOD / 6]

# Swing and stance durations
# A longer stance phase is crucial for stability and forward propulsion
SWING_PHASE_DURATION = GAIT_PERIOD / 6.0  # Duration of the swing (in the air)
STANCE_PHASE_DURATION = GAIT_PERIOD - SWING_PHASE_DURATION # Duration of the stance (on the ground)

# Neutral (standing) joint positions. The URDF model is built to be stable at zero angles,
# so we revert to this setting to ensure the hexapod can stand up.
NEUTRAL_POSITIONS = np.zeros(18)

# --- PyBullet Setup ---
def setup_pybullet_env():
    """
    Sets up the PyBullet simulation environment.
    """
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)

    p.loadURDF("plane.urdf")
    if os.path.exists(TEXTURE_PATH):
        texture_id = p.loadTexture(TEXTURE_PATH)
        p.changeVisualShape(bodyUniqueId=0, linkIndex=-1, textureUniqueId=texture_id)
    else:
        print(f"Warning: Texture file not found. Using a solid color.")

    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF file not found at: {URDF_PATH}.")
        
    start_pos = [0, 0, 0.5]
    start_ori = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(URDF_PATH, start_pos, start_ori)

    for i in range(p.getNumJoints(robot_id)):
        p.resetJointState(robot_id, i, 0.0)

    return robot_id, physicsClient

def get_keyboard_input():
    """
    Reads keyboard events from PyBullet.
    """
    keys = p.getKeyboardEvents()
    pressed_keys = {}
    for key, status in keys.items():
        if status & p.KEY_IS_DOWN:
            pressed_keys[key] = True
    return pressed_keys

def calculate_gait_target_positions(phase_counter, forward_speed, turning_speed):
    """
    Calculates the target joint positions for each leg based on the wave gait logic.
    This logic accounts for both the swing phase (in the air) and the stance phase (on the ground).
    """
    target_positions = np.zeros(18)
    
    # Iterate through each leg (0-5)
    for i in range(6):
        leg_base_joint = i * 3
        
        # Calculate the phase for the current leg, offset by its gait phase
        phase = (phase_counter + GAIT_PHASES[i]) % GAIT_PERIOD

        # Determine the side of the robot (left = 1, right = -1)
        side_multiplier = 1 if i < 3 else -1

        # Swing Phase: Leg is in the air, move to a new position
        if phase < SWING_PHASE_DURATION:
            swing_progress = phase / SWING_PHASE_DURATION
            
            # Use linear interpolation for forward/backward movement
            # and a sine wave for the vertical lift
            x_swing = FORWARD_STRIDE * forward_speed * (2.0 * swing_progress - 1.0)
            y_swing = STEP_HEIGHT * np.sin(swing_progress * np.pi)
            
            # The coxa joint controls turning
            z_swing = TURNING_STRIDE * turning_speed * (2.0 * swing_progress - 1.0) * side_multiplier
            
            target_positions[leg_base_joint] = z_swing
            target_positions[leg_base_joint + 1] = x_swing
            target_positions[leg_base_joint + 2] = -y_swing
            
        # Stance Phase: Leg is on the ground, pushing the body forward
        else:
            stance_progress = (phase - SWING_PHASE_DURATION) / STANCE_PHASE_DURATION
            
            # The foot moves backward to push the body forward
            x_stance = FORWARD_STRIDE * forward_speed * (-1.0 + 2.0 * stance_progress)
            
            # The coxa joint controls turning by sweeping sideways
            z_stance = TURNING_STRIDE * turning_speed * (2.0 * stance_progress - 1.0) * side_multiplier
            
            target_positions[leg_base_joint] = z_stance
            target_positions[leg_base_joint + 1] = x_stance
            target_positions[leg_base_joint + 2] = 0.0
            
    # Apply the neutral offset to the target positions
    target_positions += NEUTRAL_POSITIONS
            
    return target_positions

# --- Main Control Loop ---
def main():
    """
    Main function to run the manual control simulation.
    """
    print("Setting up the PyBullet environment...")
    robot_id, physicsClient = setup_pybullet_env()
    print("Setup complete. Use the following keys to control the hexapod:")
    print("Up Arrow: Forward")
    print("Down Arrow: Backward")
    print("Left Arrow: Turn Left")
    print("Right Arrow: Turn Right")
    print("Press 'q' in the PyBullet window or close the window to exit.")

    phase_counter = 0
    
    try:
        while p.isConnected(physicsClient):
            keys = get_keyboard_input()
            forward_speed = 0.0
            turning_speed = 0.0
            
            if p.B3G_UP_ARROW in keys:
                forward_speed = 1.0
            if p.B3G_DOWN_ARROW in keys:
                forward_speed = -1.0
            if p.B3G_LEFT_ARROW in keys:
                turning_speed = 1.0
            if p.B3G_RIGHT_ARROW in keys:
                turning_speed = -1.0
                
            # If any key is pressed, increment the phase counter
            if forward_speed != 0 or turning_speed != 0:
                phase_counter += 1
            else:
                # If no key is pressed, reset to a neutral state
                phase_counter = 0
                
            target_positions = calculate_gait_target_positions(phase_counter, forward_speed, turning_speed)
            
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=range(18),
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=[500] * 18
            )

            p.stepSimulation()
            time.sleep(TIME_STEP)
            
    except p.error:
        print("PyBullet connection lost, exiting.")
    finally:
        p.disconnect(physicsClient)

if __name__ == "__main__":
    main()
