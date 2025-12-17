"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from scipy.spatial.transform import Rotation

def angle_diff(target, source):
    """Compute the shortest angular difference between two angles, handling -pi/pi wrap."""
    diff = target - source
    # Wrap to [-pi, pi]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff

def normalize_angle_to_range(angle, low, high):
    """Normalize angle to be within the range [low, high], handling periodic wrap."""
    # First, wrap angle to [-pi, pi] range
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    
    # Check if we need to add/subtract 2*pi to get into range
    if angle < low:
        # Try adding 2*pi
        angle_plus_2pi = angle + 2 * np.pi
        if angle_plus_2pi <= high:
            return angle_plus_2pi
    elif angle > high:
        # Try subtracting 2*pi
        angle_minus_2pi = angle - 2 * np.pi
        if angle_minus_2pi >= low:
            return angle_minus_2pi
    
    return angle

def clip_angle_with_shortest_path(angle, low, high, reference_angle=None):
    """
    Clip angle to be within [low, high] range, finding the shortest path.
    Uses tan/arctan2 to find the shortest direction into the range.
    
    Args:
        angle: The angle to clip
        low: Lower bound of the range
        high: Upper bound of the range
        reference_angle: Optional reference angle to find shortest path from
    
    Returns:
        Clipped angle that is closest to the original angle (or reference_angle if provided)
    """
    # If already in range, return as is
    if low <= angle <= high:
        return angle
    if low<= angle - 2 *np.pi <= high:
        return angle - 2 * np.pi
    if low<= angle + 2 *np.pi <= high:
        return angle + 2 * np.pi
    
    # Generate all possible boundary values considering periodic wrap
    # For low boundary: low, low+2pi, low-2pi
    # For high boundary: high, high+2pi, high-2pi
    boundary_candidates = [
        low, low + 2 * np.pi, low - 2 * np.pi,
        high, high + 2 * np.pi, high - 2 * np.pi,
    ]
    
    # Filter to only boundaries that are actually in the [low, high] range
    valid_boundaries = [b for b in boundary_candidates if low <= b <= high]
    
    if not valid_boundaries:
        # Fallback: use low and high directly
        valid_boundaries = [low, high]
    
    # Find the boundary closest to the angle (or reference_angle if provided)
    target_angle = reference_angle if reference_angle is not None else angle
    
    best_boundary = valid_boundaries[0]
    best_candidate = angle
    min_diff = abs(angle_diff(valid_boundaries[0], target_angle))
    
    for boundary in valid_boundaries[1:]:
        diff = abs(angle_diff(boundary, target_angle))
        if diff < min_diff:
            min_diff = diff
            best_boundary = boundary
    
    return best_boundary

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name, save_video=False, classifier_queue=None):
        threading.Thread.__init__(self)
        self.queue = queue
        self.classifier_queue = classifier_queue
        self.daemon = True  # make this a daemon thread
        self.name = name
        self.save_video = save_video
        self.recording_frames = []

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if self.classifier_queue is not None:
                classifier_score = self.classifier_queue.get(timeout=1)
                if classifier_score is None:
                    classifier_score = 0
            else:
                classifier_score = 0
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (512, 512)) for k, v in img_array.items() if "full" not in k], axis=1
            )
            
            cv2.putText(frame, f"Classifier Score: {classifier_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if self.save_video:
                self.recording_frames.append(frame)

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "side": {
            "serial_number": "819312071245",
            "dim": (1280, 720),
            "exposure": 20000,
        },
        "wrist_1": {
            "serial_number": "409122274451",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP: dict[str, callable] = {
        "side": lambda img: img[0:720, 320:1040]
    }
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.ones((6,)) * 0.01
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 2
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class TZArmEnv(gym.Env):
    def __init__(
        self,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        score_display_queue=None
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], Rotation.from_euler("XYZ", config.RESET_POSE[3:]).as_quat()]
        )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        # "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "q": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "dq": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "effort": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        requests.post(self.url + "/api/controller/start")
        requests.post(self.url + "/api/controller/realtime/start")
        time.sleep(2)
        self._update_currpos()

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url, classifier_queue=score_display_queue)
            self.displayer.start()

        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized TuringZero Arm")

    def angle_diff(self,target, source):
        """Compute the shortest angular difference between two angles, handling -pi/pi wrap."""
        diff = target - source
        # Wrap to [-pi, pi]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return diff

    def clip_safety_box(self, action: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        nextpos = self.currpos.copy()
        # get target pose from action and current pose
        xyz_delta = action[:3] / 1000.0 * 60 * 0.02 # 需要小于60
        next_xyz = nextpos[:3] + xyz_delta 

        curr_rot = Rotation.from_quat(self.currpos[3:])
        curr_rpy = curr_rot.as_euler("XYZ")
        for i in range(3):
            curr_rpy[i] = normalize_angle_to_range(curr_rpy[i], self.rpy_bounding_box.low[i], self.rpy_bounding_box.high[i])

        next_rot = Rotation.from_rotvec(action[3:6]) * curr_rot
        
        # Clip xyz
        next_xyz = np.clip(
            next_xyz, self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        
        # Clip rpy angles, handling periodic wrap
        # Use shortest path based on current angle (curr_rpy) as reference
        next_rpy = next_rot.as_euler("XYZ")
        for i in range(3):
            next_rpy[i] = clip_angle_with_shortest_path(
                next_rpy[i], 
                self.rpy_bounding_box.low[i], 
                self.rpy_bounding_box.high[i],
                reference_angle=curr_rpy[i]
            )

        # Pitch (index 1) doesn't have -pi/pi discontinuity in typical ranges
        next_rpy[1] = clip_angle_with_shortest_path(
            next_rpy[1], 
            self.rpy_bounding_box.low[1], 
            self.rpy_bounding_box.high[1],
            reference_angle=curr_rpy[1]
        )
        
        next_rot = Rotation.from_euler("XYZ", next_rpy)
        diff_rot = next_rot * curr_rot.inv()
        diff_euler = diff_rot.as_euler("XYZ")
        
        # Compute the actual delta after clipping, using angle_diff to handle periodic wrap
        xyz_delta_clipped = next_xyz - self.currpos[:3]
        rpy_delta_clipped = diff_euler

        # Convert back to action space
        action[:3] = xyz_delta_clipped * 1000.0 / 60.0 / 0.02
        action[3:6] = rpy_delta_clipped / (0.5 * 0.02)

        return action[:6]

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # gripper_action = action[6]
        action[:6] = action[:6] * self.action_scale[:6]
        action[:6] = self.clip_safety_box(action[:6])
        # self._send_gripper_command(gripper_action)
        self._send_delta_pos_command(action[:6])

        self.curr_path_length += 1

        self._update_currpos()
        currpos = self.currpos.copy()
        currpos = np.concatenate([currpos[:3], Rotation.from_quat(currpos[3:]).as_euler("XYZ")])
        ob = self._get_obs()
        # reward = self.compute_reward(ob)
        reward = 0
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate

        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("XYZ", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("XYZ")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = copy.deepcopy(resized)
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = Rotation.from_euler("XYZ", euler_random).as_quat()
            self._send_ptp_command(reset_pose, timeout=1.5)
        else:
            reset_pose = self.resetpos.copy()
            self._send_ptp_command(reset_pose, timeout=1.5)

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], Rotation.from_euler("XYZ", goal[3:]).as_quat()])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_ptp_command(p, timeout=1)
        self._update_currpos()

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.displayer.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.displayer.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.displayer.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.displayer.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.displayer.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        pass

    def _send_ptp_command(self, pos: np.ndarray, timeout: float):
        """Internal function to send position command to the robot."""
        # self._recover()
        if pos.shape == (7,):
            pos = np.concatenate([pos[:3], Rotation.from_quat(pos[3:]).as_euler("xyz")])

        response = requests.post(self.url + "/api/controller/ptp", json={
            "position": pos[:3].tolist(),
            "rpy": pos[3:].tolist(),
            "exec_time": timeout
        })
        if response.status_code != 200:
            print(f"Failed to send PTP command: {response.json()}")
            return
        time.sleep(timeout + 2)

    def _send_line_command(self, pos: np.ndarray, timeout: float):
        """Internal function to send line command to the robot."""
        response = requests.post(self.url + "/api/controller/line", json={
            "position": pos[:3].tolist(),
            "rpy": pos[3:].tolist(),
            "exec_time": timeout
        })
        if response.status_code != 200:
            print(f"Failed to send line command: {response.json()}")
            return
        time.sleep(timeout + 2)

    def _send_delta_pos_command(self, dpos: np.ndarray):
        arr = np.array(dpos).astype(np.float32).tolist()
        data = {"x": arr[0], "y": arr[1], "z": arr[2]}
        requests.post(self.url + "/api/control/position", json=data)
        data = {"r": arr[3], "p": arr[4], "yaw": arr[5]}
        requests.post(self.url + "/api/control/rpy", json=data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        print(f"curr_gripper_pos: {self.curr_gripper_is_closed}")
        if mode == "binary":
            if (pos <= 0.5):  # close gripper
                response = requests.post(self.url + "/api/gripper/close", json={"force": 1.0})
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5):  # open gripper
                response = requests.post(self.url + "/api/gripper/open", json={"force": -0.5})
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
                # requests.post(self.url + "/api/gripper/release")
                # time.sleep(self.gripper_sleep)
            else: 
                return
            if response.status_code != 200:
                print(f"Failed to send gripper command: {response.json()}")
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.get(self.url + "/api/status").json()['data']
        tcp_pose = ps["tcp_pose"]
        tcp_velocity = ps["tcp_velocity"]

        self.currpos = np.concatenate([tcp_pose["position"], tcp_pose["rpy"]])
        if self.currpos.shape == (6,):
            self.currpos = np.concatenate([self.currpos[:3], Rotation.from_euler("XYZ", self.currpos[3:]).as_quat()])
        self.currvel = np.concatenate([tcp_velocity["linear"], tcp_velocity["angular"]])

        self.currq = np.array(ps["joints"]["positions"])
        self.currqvel = np.array(ps["joints"]["rpm"])
        self.currtorque = np.array(ps["joints"]["current"]) * 100.0

        self.curr_gripper_is_closed = ps["gripper"]["is_closed"]
        # self.curr_gripper_effort = np.array(ps["gripper"]["effort"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            # "gripper_pose": self.curr_gripper_pos,
            "q": self.currq,
            "dq": self.currqvel,
            "effort": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put((None, None))
            cv2.destroyAllWindows()
            self.displayer.join()
