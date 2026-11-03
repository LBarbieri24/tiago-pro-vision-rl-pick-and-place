#!/usr/bin/env python3
"""
collect_dataset.py - Automatically generates a labeled dataset of a randomized block and table in Gazebo.
It spawns the table and cube with randomized colors and sizes at randomized positions on the table,
projects the 3D corners of the actual settled model onto the camera plane to compute the exact 2D bounding box,
saves the images/labels in 'dataset_v2', and partitions them into train/val folders for YOLOv8 training.
"""

import sys
import os
import math
import random
import time
import subprocess
import cv2
import numpy as np
import site

# Add ROS 2 and system paths
ros_paths = [
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
    "/opt/pal/alum/lib/python3.10/site-packages",
    "/home/user/.local/lib/python3.10/site-packages"
]
for p in ros_paths:
    if p not in sys.path:
        sys.path.append(p)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class DatasetCollectorNode(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Intrinsics
        self.intrinsics = None
        self.camera_frame = None
        
        # Latest image
        self.latest_cv_img = None
        self.latest_img_stamp = None
        
        # Service clients
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        
        # Action client for head control
        self.head_client = ActionClient(self, FollowJointTrajectory, '/head_controller/follow_joint_trajectory')
        # Action client for torso control
        self.torso_client = ActionClient(self, FollowJointTrajectory, '/torso_controller/follow_joint_trajectory')
        
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/head_front_camera/color/camera_info',
            self.camera_info_callback,
            qos
        )
        
        self.img_sub = self.create_subscription(
            Image,
            '/head_front_camera/color/image_raw',
            self.image_callback,
            qos
        )

    def camera_info_callback(self, msg):
        # Extract camera intrinsics K
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        self.intrinsics = (fx, fy, cx, cy)
        self.camera_frame = msg.header.frame_id

    def image_callback(self, msg):
        try:
            self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_img_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")

    def move_head_and_torso(self, head_1, head_2, torso_lift, duration_sec=1.0):
        head_ok = self.head_client.wait_for_server(timeout_sec=5.0)
        torso_ok = self.torso_client.wait_for_server(timeout_sec=5.0)
        if not (head_ok and torso_ok):
            self.get_logger().error("Head or Torso controller action server NOT available!")
            return False
            
        # 1. Build head goal
        head_goal = FollowJointTrajectory.Goal()
        head_goal.trajectory.joint_names = ['head_1_joint', 'head_2_joint']
        head_p = JointTrajectoryPoint()
        head_p.positions = [float(head_1), float(head_2)]
        head_p.time_from_start = Duration(sec=0, nanosec=int(duration_sec * 1e9))
        head_goal.trajectory.points = [head_p]
        
        # 2. Build torso goal
        torso_goal = FollowJointTrajectory.Goal()
        torso_goal.trajectory.joint_names = ['torso_lift_joint']
        torso_p = JointTrajectoryPoint()
        torso_p.positions = [float(torso_lift)]
        torso_p.time_from_start = Duration(sec=0, nanosec=int(duration_sec * 1e9))
        torso_goal.trajectory.points = [torso_p]
        
        # 3. Send both goals concurrently
        head_future = self.head_client.send_goal_async(head_goal)
        torso_future = self.torso_client.send_goal_async(torso_goal)
        
        rclpy.spin_until_future_complete(self, head_future, timeout_sec=2.0)
        rclpy.spin_until_future_complete(self, torso_future, timeout_sec=2.0)
        
        head_handle = head_future.result()
        torso_handle = torso_future.result()
        
        if head_handle is None or not head_handle.accepted or torso_handle is None or not torso_handle.accepted:
            self.get_logger().error("Head or Torso goal was rejected!")
            return False
            
        # 4. Wait for both results
        head_res_future = head_handle.get_result_async()
        torso_res_future = torso_handle.get_result_async()
        
        rclpy.spin_until_future_complete(self, head_res_future, timeout_sec=duration_sec + 1.0)
        rclpy.spin_until_future_complete(self, torso_res_future, timeout_sec=duration_sec + 1.0)
        return True

    def _gz_poll_robot_pose(self) -> tuple[float, float, float, float] | None:
        for _ in range(3):
            try:
                result = subprocess.run(
                    ["gz", "model", "-m", "tiago-pro", "-p"],
                    text=True,
                    timeout=2.0,
                    capture_output=True
                )
                parts = result.stdout.strip().split()
                if len(parts) >= 6:
                    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[5])
            except Exception:
                pass
            time.sleep(0.05)
        return None

    def _gz_poll_model(self, model_name: str) -> tuple[float, float, float, float] | None:
        for _ in range(3):
            try:
                result = subprocess.run(
                    ["gz", "model", "-m", model_name, "-p"],
                    text=True,
                    timeout=2.0,
                    capture_output=True
                )
                parts = result.stdout.strip().split()
                if len(parts) >= 6:
                    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[5])
            except Exception:
                pass
            time.sleep(0.05)
        return None

    def delete_cube(self):
        if not self.delete_client.wait_for_service(timeout_sec=2.0):
            return False
        req = DeleteEntity.Request()
        req.name = "cube"
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        return res is not None and res.success

    def spawn_cube(self, x, y, z, yaw, sdf_str):
        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            return False
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)
        
        req = SpawnEntity.Request()
        req.name = "cube"
        req.xml = sdf_str
        req.initial_pose = pose
        req.reference_frame = "world"
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        return res is not None and res.success

    def delete_table(self):
        if not self.delete_client.wait_for_service(timeout_sec=2.0):
            return False
        req = DeleteEntity.Request()
        req.name = "table"
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        return res is not None and res.success

    def spawn_table(self, x, y, z, yaw, sdf_str):
        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            return False
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)
        
        req = SpawnEntity.Request()
        req.name = "table"
        req.xml = sdf_str
        req.initial_pose = pose
        req.reference_frame = "world"
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        return res is not None and res.success

def _material(r, g, b, a=1.0) -> str:
    return f"""
          <material>
            <ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>
            <diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>"""

def build_table_sdf(r, g, b, table_height=0.5) -> str:
    TABLE_SURFACE_HEIGHT = 0.03
    TABLE_SURFACE_WIDTH = 0.70
    TABLE_SURFACE_DEPTH = 0.70
    TABLE_LEG_RADIUS = 0.03
    TABLE_MASS = 20.0

    leg_height = table_height - TABLE_SURFACE_HEIGHT
    leg_z      = leg_height / 2.0
    top_z      = leg_height + TABLE_SURFACE_HEIGHT / 2.0

    hw = TABLE_SURFACE_WIDTH  / 2.0
    hd = TABLE_SURFACE_DEPTH  / 2.0
    lr = TABLE_LEG_RADIUS

    legs = [
        (+hw - lr, +hd - lr),
        (+hw - lr, -hd + lr),
        (-hw + lr, +hd - lr),
        (-hw + lr, -hd + lr),
    ]

    leg_links = ""
    for i, (lx, ly) in enumerate(legs):
        leg_links += f"""
      <link name="leg_{i}">
        <pose>{lx} {ly} {leg_z} 0 0 0</pose>
        <inertial>
          <mass>{TABLE_MASS / 5.0}</mass>
        </inertial>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>{lr}</radius>
              <length>{leg_height}</length>
            </cylinder>
          </geometry>{_material(r * 0.7, g * 0.7, b * 0.7)}
        </visual>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>{lr}</radius>
              <length>{leg_height}</length>
            </cylinder>
          </geometry>
        </collision>
      </link>
      <joint name="leg_{i}_joint" type="fixed">
        <parent>tabletop</parent>
        <child>leg_{i}</child>
      </joint>"""

    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="table">
    <static>true</static>
    <link name="tabletop">
      <pose>0 0 {top_z} 0 0 0</pose>
      <inertial>
        <mass>{TABLE_MASS}</mass>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>{TABLE_SURFACE_WIDTH} {TABLE_SURFACE_DEPTH} {TABLE_SURFACE_HEIGHT}</size>
          </box>
        </geometry>{_material(r, g, b)}
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>{TABLE_SURFACE_WIDTH} {TABLE_SURFACE_DEPTH} {TABLE_SURFACE_HEIGHT}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.8</mu>
              <mu2>0.8</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
{leg_links}
  </model>
</sdf>
"""

def build_cube_sdf(size_x, size_y, size_z, r, g, b):
    CUBE_MASS = 0.01 * (size_x * size_y * size_z) / (0.04 * 0.04 * 0.075)
    ix = CUBE_MASS / 12.0 * (size_y**2 + size_z**2)
    iy = CUBE_MASS / 12.0 * (size_x**2 + size_z**2)
    iz = CUBE_MASS / 12.0 * (size_x**2 + size_y**2)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="cube">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{CUBE_MASS:.6f}</mass>
        <inertia>
          <ixx>{ix:.8f}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{iy:.8f}</iyy>
          <iyz>0.0</iyz>
          <izz>{iz:.8f}</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>{size_x:.4f} {size_y:.4f} {size_z:.4f}</size>
          </box>
        </geometry>{_material(r, g, b)}
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>{size_x:.4f} {size_y:.4f} {size_z:.4f}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
  </model>
</sdf>
"""

def wait_for_new_image(node, t_spawn, wait_duration=0.5):
    t_spawn_ns = t_spawn.nanoseconds
    target_ns = t_spawn_ns + int(wait_duration * 1e9)
    start_wall_time = time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        if node.latest_img_stamp is not None:
            img_ns = node.latest_img_stamp.sec * 10**9 + node.latest_img_stamp.nanosec
            if img_ns >= target_ns:
                rclpy.spin_once(node, timeout_sec=0.01)
                break
        
        # Fallback if simulator is paused or time is not moving
        if time.time() - start_wall_time > 5.0:
            print("Warning: Wait for new simulation image timed out. Proceeding.")
            break

def collect_data(node, num_images=1500):
    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_dir = os.path.join(ws_root, "datasets", "dataset_v2")
    
    for sub in ["images/train", "images/val", "labels/train", "labels/val", "debug"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
        
    # Write dataset.yaml
    yaml_content = f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

names:
  0: block
"""
    with open(os.path.join(base_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"Directories and dataset.yaml prepared in '{base_dir}'.")
    print("Waiting for camera info and image topics...")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.intrinsics is not None and node.latest_cv_img is not None:
            break
        time.sleep(0.5)
    print("Topics active!")
    
    # Find the maximum existing frame index to avoid overwriting
    existing_indices = []
    for split in ["train", "val"]:
        img_dir = os.path.join(base_dir, "images", split)
        if os.path.exists(img_dir):
            for filename in os.listdir(img_dir):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    try:
                        idx = int(filename[6:-4])
                        existing_indices.append(idx)
                    except ValueError:
                        pass
    
    count = max(existing_indices) + 1 if existing_indices else 0
    start_count = count
    target_count = count + num_images
    print(f"Resuming/starting dataset collection from frame index: {count}")
    print(f"Will collect {num_images} images (stopping at frame_{target_count:04d})")
    
    failed_tries = 0
    
    # Persistent table variables
    tr, tg, tb = 0.05, 0.05, 0.05
    table_height = 0.5
    
    while count < target_count and rclpy.ok():
        robot_pose = node._gz_poll_robot_pose()
        if robot_pose is None:
            print("Could not get robot pose, retrying...")
            time.sleep(1.0)
            continue
            
        rx, ry, rz, ryaw = robot_pose
        
        # Move head and torso to random positions (head +-0.2 rad, torso +-0.1 m from nominal 0.1 m)
        h1 = random.uniform(-0.2, 0.2)
        h2 = -0.785398 + random.uniform(-0.2, 0.2)
        t_lift = random.uniform(0.0, 0.2)
        node.move_head_and_torso(h1, h2, t_lift, duration_sec=1.0)
        
        # If we need to change table color and height (every 15 frames, or first iteration of this run)
        if count == start_count or count % 15 == 0:
            tr, tg, tb = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
            table_height = random.uniform(0.42, 0.58)
            
        cr, cg, cb = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
        
        # Ensure cube color is reasonably distinct from table color
        color_diff = math.sqrt((cr-tr)**2 + (cg-tg)**2 + (cb-tb)**2)
        if color_diff < 0.25:
            continue
            
        # Randomize cube sizes
        size_x = random.uniform(0.03, 0.08)
        size_y = random.uniform(0.03, 0.08)
        size_z = random.uniform(0.05, 0.12)
        
        # Randomize cube offset on table (table centered at TABLE_POS_X=0.75 relative to robot)
        offset_x = random.uniform(-0.23, 0.23)
        offset_y = random.uniform(-0.23, 0.23)
        cube_yaw = random.uniform(0, 2.0 * math.pi)
        
        TABLE_POS_X = 0.75
        TABLE_POS_Y = 0.0
        
        cube_dx = TABLE_POS_X + offset_x
        cube_dy = TABLE_POS_Y + offset_y
        
        # Add 5mm spawn offset above table height to prevent violent collision glitches
        cube_dz = table_height + size_z / 2.0 + 0.005
        
        cube_x = rx + cube_dx * math.cos(ryaw) - cube_dy * math.sin(ryaw)
        cube_y = ry + cube_dx * math.sin(ryaw) + cube_dy * math.cos(ryaw)
        cube_z = rz + cube_dz
        
        table_x = rx + TABLE_POS_X * math.cos(ryaw) - TABLE_POS_Y * math.sin(ryaw)
        table_y = ry + TABLE_POS_X * math.sin(ryaw) + TABLE_POS_Y * math.cos(ryaw)
        table_z = rz
        table_yaw = ryaw
        
        cube_sdf = build_cube_sdf(size_x, size_y, size_z, cr, cg, cb)
        
        # Delete cube
        node.delete_cube()
        time.sleep(0.5)
        
        # Only rebuild table every 15 frames to prevent Gazebo crash, or first iteration of this run
        if count == start_count or count % 15 == 0:
            table_sdf = build_table_sdf(tr, tg, tb, table_height)
            node.delete_table()
            time.sleep(1.0)
            for _ in range(3):
                if node.spawn_table(table_x, table_y, table_z, table_yaw, table_sdf):
                    break
                time.sleep(0.8)
            
        # Spawn cube with retries
        spawn_success = False
        for _ in range(3):
            if node.spawn_cube(cube_x, cube_y, cube_z, cube_yaw, cube_sdf):
                spawn_success = True
                break
            time.sleep(0.8)
            
        if not spawn_success:
            print("Warning: Failed to spawn cube after retries, skipping...")
            continue
            
        t_spawn = node.get_clock().now()
        
        # Wait for camera frame and physics settlement (1.5s sim time)
        wait_for_new_image(node, t_spawn, wait_duration=1.5)
        
        # Poll the actual cube pose after physics settling
        actual_cube = node._gz_poll_model("cube")
        if actual_cube is None:
            print("Warning: Could not poll cube pose from Gazebo, skipping...")
            continue
            
        acx, acy, acz, acyaw = actual_cube
        
        # Verify if the cube has fallen off the table or glitched away
        if acz < (table_height - 0.05) or acz > (table_height + size_z + 0.1):
            print(f"Warning: Cube fell off table or glitched (z={acz:.3f}), skipping...")
            continue
            
        # Compute exact pose in base_footprint based on real polled pose
        dx = acx - rx
        dy = acy - ry
        dz = acz - rz
        
        cos_ryaw = math.cos(-ryaw)
        sin_ryaw = math.sin(-ryaw)
        x_base = dx * cos_ryaw - dy * sin_ryaw
        y_base = dx * sin_ryaw + dy * cos_ryaw
        z_base = dz
        yaw_base = acyaw - ryaw
        
        # 8 local corners of the cube bounding box using the randomized size
        corners_local = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corners_local.append((sx * size_x / 2.0, sy * size_y / 2.0, sz * size_z / 2.0))
                    
        # Transform local corners to base_footprint
        cos_cy = math.cos(yaw_base)
        sin_cy = math.sin(yaw_base)
        corners_base = []
        for lx, ly, lz in corners_local:
            bx = x_base + lx * cos_cy - ly * sin_cy
            by = y_base + lx * sin_cy + ly * cos_cy
            bz = z_base + lz
            corners_base.append((bx, by, bz))
            
        # Get TF transform from base_footprint to camera at the exact image timestamp
        try:
            lookup_time = rclpy.time.Time(seconds=node.latest_img_stamp.sec, nanoseconds=node.latest_img_stamp.nanosec)
            t_msg = node.tf_buffer.lookup_transform(node.camera_frame, 'base_footprint', lookup_time)
        except Exception as e:
            print(f"TF Lookup failed: {e}")
            failed_tries += 1
            time.sleep(0.5)
            continue
            
        # Project 3D corners to 2D image coordinates
        fx, fy, cx, cy_intr = node.intrinsics
        u_list = []
        v_list = []
        
        trans = t_msg.transform.translation
        rot = t_msg.transform.rotation
        qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
        
        # Quaternion to rotation matrix
        r00 = 1.0 - 2.0 * (qy**2 + qz**2)
        r01 = 2.0 * (qx*qy - qz*qw)
        r02 = 2.0 * (qx*qz + qy*qw)
        
        r10 = 2.0 * (qx*qy + qz*qw)
        r11 = 1.0 - 2.0 * (qx**2 + qz**2)
        r12 = 2.0 * (qy*qz - qx*qw)
        
        r20 = 2.0 * (qx*qz - qy*qw)
        r21 = 2.0 * (qy*qz + qx*qw)
        r22 = 1.0 - 2.0 * (qx**2 + qy**2)
        
        for bx, by, bz in corners_base:
            cam_x = r00 * bx + r01 * by + r02 * bz + trans.x
            cam_y = r10 * bx + r11 * by + r12 * bz + trans.y
            cam_z = r20 * bx + r21 * by + r22 * bz + trans.z
            
            if cam_z <= 0.1:
                continue
            u = fx * (cam_x / cam_z) + cx
            v = fy * (cam_y / cam_z) + cy_intr
            u_list.append(u)
            v_list.append(v)
            
        if len(u_list) < 8:
            print("Warning: Some corners are behind camera, skipping...")
            continue
            
        # Bounding box
        u_min, u_max = min(u_list), max(u_list)
        v_min, v_max = min(v_list), max(v_list)
        
        img = node.latest_cv_img.copy()
        img_h, img_w, _ = img.shape
        
        margin = 5
        if u_min < margin or u_max > img_w - margin or v_min < margin or v_max > img_h - margin:
            # Partly out of bounds, skip
            continue
            
        bbox_w = u_max - u_min
        bbox_h = v_max - v_min
        if bbox_w < 12 or bbox_h < 12:
            # Too small, skip
            continue
            
        # Save frame
        is_val = (count % 5 == 0)
        split = "val" if is_val else "train"
        
        x_cnt = ((u_min + u_max) / 2.0) / img_w
        y_cnt = ((v_min + v_max) / 2.0) / img_h
        norm_w = bbox_w / img_w
        norm_h = bbox_h / img_h
        
        # Write annotation
        label_path = os.path.join(base_dir, "labels", split, f"frame_{count:04d}.txt")
        with open(label_path, "w") as lf:
            lf.write(f"0 {x_cnt:.6f} {y_cnt:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
        # Save image
        img_path = os.path.join(base_dir, "images", split, f"frame_{count:04d}.jpg")
        cv2.imwrite(img_path, node.latest_cv_img)
        
        # Save debug image
        dbg_img = node.latest_cv_img.copy()
        cv2.rectangle(dbg_img, (int(u_min), int(v_min)), (int(u_max), int(v_max)), (0, 255, 0), 2)
        cv2.putText(dbg_img, "block", (int(u_min), int(v_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(base_dir, "debug", f"frame_{count:04d}.jpg"), dbg_img)
        
        print(f"✓ Saved frame_{count:04d} ({split}) - Bbox: [{int(u_min)}, {int(v_min)}, {int(u_max)}, {int(v_max)}]")
        count += 1
        time.sleep(0.5)

    # Reset table and cube to neutral default positions and colors
    node.delete_cube()
    time.sleep(0.5)
    node.delete_table()
    time.sleep(1.0)
    
    # Neutral Colors
    neutral_table_sdf = build_table_sdf(0.05, 0.05, 0.05, table_height=0.5)
    neutral_cube_sdf = build_cube_sdf(0.04, 0.04, 0.075, 0.9, 0.1, 0.1)
    
    node.spawn_table(table_x, table_y, table_z, table_yaw, neutral_table_sdf)
    time.sleep(0.5)
    
    default_x = rx + TABLE_POS_X * math.cos(ryaw) - TABLE_POS_Y * math.sin(ryaw)
    default_y = ry + TABLE_POS_X * math.sin(ryaw) + TABLE_POS_Y * math.cos(ryaw)
    default_z = rz + 0.5 + 0.075 / 2.0
    
    node.spawn_cube(default_x, default_y, default_z, ryaw, neutral_cube_sdf)
    
    # Reset head and torso to neutral default positions
    node.move_head_and_torso(0.0, -0.785398, 0.0, duration_sec=1.5)
    
    print("Table, Cube and Head reset to default positions and neutral colors.")
    print("Dataset collection complete!")

def main():
    rclpy.init()
    node = DatasetCollectorNode()
    try:
        collect_data(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
