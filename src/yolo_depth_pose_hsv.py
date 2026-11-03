#!/usr/bin/env python3
import sys
import os
import math
import argparse
import threading
import queue
import time
import numpy as np
import cv2

import site

# Sourcing system packages to import ROS 2 modules in custom environment
ros_paths = [
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
    "/opt/pal/alum/lib/python3.10/site-packages",
    "/home/user/.local/lib/python3.10/site-packages",  # Docker container user packages (torch, clip, ultralytics)
    site.getusersitepackages()
]
for p in ros_paths:
    if p not in sys.path:
        sys.path.append(p)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header, Bool

# Optional YOLO import
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from filters import ExponentialMovingAverage, ExponentialMovingAverageYaw

# EMA Filter Alpha Parameters
EMA_POS_ALPHA_DEFAULT = 0.40
EMA_POS_ALPHA_NEAR_EE = 1.0
EE_NEAR_THRESHOLD_M = 0.08

EMA_YAW_ALPHA_DEFAULT = 0.01
EMA_YAW_ALPHA_LOCKED = 0.0


def wrap_to_pi_over_2(angle):
    return (angle + math.pi / 2) % math.pi - math.pi / 2








class YoloDepthPoseNode(Node):
    def __init__(self, yolo_model_path, yolo_conf, parent_frame,
                 depth_topic,
                 max_depth_m=2.5, pose_offset=[0.0, 0.0, 0.0],
                 goal_position=[0.75, 0.0, 0.7], hsv_thresh=[100, 100, 80],
                 debug_save_dir=None, fixed_yaw_deg=None, show_obb=True):
        super().__init__('yolo_depth_pose_node')
        
        self.get_logger().info("Tiago Pro deployment mode.")
        
        self.yolo_conf = yolo_conf
        self.parent_frame = parent_frame
        self.depth_topic = depth_topic
        self.pose_offset = pose_offset
        self.goal_position = [float(x) for x in goal_position]
        self.get_logger().info(f"Goal position configured as: {self.goal_position}")
        self.max_depth_m = max_depth_m
        
        self.hsv_h_thresh, self.hsv_s_thresh, self.hsv_v_thresh = hsv_thresh
        self.show_obb = show_obb

        # Fixed yaw override: if set, replaces detected yaw with a constant value
        if fixed_yaw_deg is not None:
            self.fixed_yaw_rad = math.radians(fixed_yaw_deg)
            self.get_logger().info(f"Fixed yaw override: {fixed_yaw_deg:.1f} deg ({self.fixed_yaw_rad:.4f} rad)")
        else:
            self.fixed_yaw_rad = None

        # Debug snapshot directory (None = disabled)
        self.debug_save_dir = None
        if debug_save_dir is not None:
            self.debug_save_dir = os.path.expanduser(debug_save_dir)
            os.makedirs(self.debug_save_dir, exist_ok=True)
            self.get_logger().info(f"Asynchronous debug snapshots will be saved to: {self.debug_save_dir}")
            
        # Start background saving/visualization thread unconditionally
        self.save_queue = queue.Queue(maxsize=100)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()



        

        
        # Load YOLO if requested and available
        self.yolo_model = None
        if HAS_ULTRALYTICS:
            self.get_logger().info(f"Loading YOLO model from {yolo_model_path}...")
            try:
                self.yolo_model = YOLO(yolo_model_path)
                self.get_logger().info("YOLO model loaded successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
                sys.exit(1)
        else:
            self.get_logger().error("Ultralytics/YOLO is not installed. Exiting.")
            sys.exit(1)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Intrinsics state
        self.intrinsics = None  # (fx, fy, cx, cy)
        self.camera_frame = None
        self.last_predicted_pose = None
        self.last_predicted_yaw = 0.0

        # EMA filters
        self.ema_x = None
        self.ema_y = None
        self.ema_z = None
        self.ema_yaw = None
        self.yaw_locked = False
        # 2-seconds yaw calibration buffer
        self.yaw_calib_buffer = []       # raw theta values during calibration window
        self.yaw_calib_start = None      # wall-clock time when calibration started

        # Synced latest depth image
        self.latest_depth_msg = None
        self.latest_depth_lock = threading.Lock()

        self.is_grasped = False

        pub_specs = [
            ('pose_pub', PoseStamped, '/predicted_object_pose'),
            ('vis_pub', Image, '/head_front_camera/color/image_with_yolo'),
            ('marker_pub', Marker, '/predicted_object_marker'),
            ('fitted_rect_pub', Marker, '/fitted_rectangle_marker'),
            ('goal_marker_pub', Marker, '/goal_marker'),
            ('goal_pub', PoseStamped, '/goal_position'),
            ('ee_marker_pub', Marker, '/ee_offset_marker'),
        ]
        for attr, msg_type, topic in pub_specs:
            setattr(self, attr, self.create_publisher(msg_type, topic, 10))

        # Periodically publish goal marker and position at 2 Hz
        self.goal_marker_timer = self.create_timer(0.5, self.publish_goal_marker)
        

        # Subscriptions
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

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos
        )

        self.img_sub = self.create_subscription(
            Image,
            '/head_front_camera/color/image_raw',
            self.image_callback,
            qos
        )

        grasped_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.grasped_sub = self.create_subscription(
            Bool,
            '/grasped_current',
            self.grasped_callback,
            grasped_qos
        )

        self.get_logger().info("Node initialized. Mode: YOLO + HSV ROI preprocessing")
        self.get_logger().info(f"Subscribed to depth on topic: {self.depth_topic}")
        self._dbg_counter = 0

    def camera_info_callback(self, msg):
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        self.intrinsics = (fx, fy, cx, cy)
        self.camera_frame = msg.header.frame_id

    def grasped_callback(self, msg: Bool):
        self.is_grasped = bool(msg.data)

    def depth_callback(self, msg):
        with self.latest_depth_lock:
            self.latest_depth_msg = msg



    def detect_object_yolo(self, image):
        if self.yolo_model is None:
            return None, 0.0
        results = self.yolo_model(image, verbose=False)[0]
        
        h, w, _ = image.shape
        img_area = w * h
        max_area = 0.25 * img_area
        
        if len(results.boxes) == 0:
            self.get_logger().info("YOLO: No objects detected in the image.", throttle_duration_sec=10.0)
            return None, 0.0

        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.yolo_model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            
            self.get_logger().info(
                f"YOLO: Detected '{cls_name}' with conf {conf:.2f} (area: {area}/{img_area} px, {area/img_area*100:.1f}%)",
                throttle_duration_sec=10.0
            )
            
            if area <= max_area and conf > self.yolo_conf and conf > best_conf:
                best_box = (x1, y1, x2, y2)
                best_conf = conf

        return best_box, best_conf

    def transform_to_matrix(self, trans):
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        tz = trans.transform.translation.z
        
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w
        
        r00 = 1.0 - 2.0 * (qy**2 + qz**2)
        r01 = 2.0 * (qx*qy - qz*qw)
        r02 = 2.0 * (qx*qz + qy*qw)
        
        r10 = 2.0 * (qx*qy + qz*qw)
        r11 = 1.0 - 2.0 * (qx**2 + qz**2)
        r12 = 2.0 * (qy*qz - qx*qw)
        
        r20 = 2.0 * (qx*qz - qy*qw)
        r21 = 2.0 * (qy*qz + qx*qw)
        r22 = 1.0 - 2.0 * (qx**2 + qy**2)
        
        T = np.array([
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0.0, 0.0, 0.0, 1.0]
        ])
        return T

    def _make_marker(self, stamp, ns, mid, mtype, position=None, scale=(0.03, 0.03, 0.03),
                     color=(0.0, 1.0, 0.0, 0.6), lifetime=0.0, orientation=(0.0, 0.0, 0.0, 1.0)):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.parent_frame
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        if position is not None:
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(position[0]), float(position[1]), float(position[2])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = orientation
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        if lifetime > 0:
            m.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()
        return m

    def _publish_pose_and_tf(self, stamp, px, py, pz, theta, is_lost=False, is_grasped=False):
        qx, qy, qz, qw = 0.0, 0.0, math.sin(theta * 0.5), math.cos(theta * 0.5)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.parent_frame
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = px, py, pz
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y = qx, qy
        pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = qz, qw
        self.pose_pub.publish(pose_msg)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = 'predicted_object_frame'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = px, py, pz
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self.tf_broadcaster.sendTransform(t)

        self._publish_marker(stamp, px, py, pz, theta, is_lost=is_lost, is_grasped=is_grasped)

    def handle_detection_failure(self, header, cv_img=None):
        if self.is_grasped:
            self._publish_grasped_ee_pose(header, cv_img)
            return
        self.get_logger().warn("Detection lost. Publishing last known prediction.", throttle_duration_sec=3.0)
        self._publish_last_prediction(header, cv_img)

    def _publish_grasped_ee_pose(self, header, cv_img=None):
        stamp = header.stamp
        px, py, pz, theta = 0.0, 0.0, 0.0, 0.0
        try:
            trans_ee = self.tf_buffer.lookup_transform(
                self.parent_frame,
                'gripper_right_grasping_link',
                rclpy.time.Time()
            )
            px = trans_ee.transform.translation.x
            py = trans_ee.transform.translation.y
            pz = trans_ee.transform.translation.z
            qx = trans_ee.transform.rotation.x
            qy = trans_ee.transform.rotation.y
            qz = trans_ee.transform.rotation.z
            qw = trans_ee.transform.rotation.w
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            theta = math.atan2(siny_cosp, cosy_cosp)
            if self.fixed_yaw_rad is not None:
                theta = self.fixed_yaw_rad
            self.last_predicted_pose = (px, py, pz)
            self.last_predicted_yaw = theta
        except Exception:
            if self.last_predicted_pose is not None:
                px, py, pz = self.last_predicted_pose
                theta = self.last_predicted_yaw

        self._publish_pose_and_tf(stamp, px, py, pz, theta, is_lost=False, is_grasped=True)
        if cv_img is not None:
            self.queue_visualization_task(
                header=header, cv_img=cv_img, px=px, py=py, pz=pz, theta=theta,
                box_3d_val=None, is_lost=False, is_grasped=True, status_text="GRASPED (EE Pose)"
            )

    def _publish_last_prediction(self, header, cv_img=None):
        stamp = header.stamp
        if self.last_predicted_pose is None:
            if cv_img is not None:
                self.queue_visualization_task(
                    header=header, cv_img=cv_img, px=None, py=None, pz=None, theta=None,
                    box_3d_val=None, is_lost=True, is_grasped=self.is_grasped, status_text="LOST"
                )
            return
        px, py, pz = self.last_predicted_pose
        theta = self.last_predicted_yaw

        self._publish_pose_and_tf(stamp, px, py, pz, theta, is_lost=True, is_grasped=self.is_grasped)

        self.queue_visualization_task(
            header=header, cv_img=cv_img, px=px, py=py, pz=pz, theta=theta,
            box_3d_val=None, is_lost=True, is_grasped=self.is_grasped, status_text="LOST"
        )

    def publish_goal_marker(self):
        stamp = self.get_clock().now().to_msg()
        marker = self._make_marker(stamp, "goal_position", 1, Marker.SPHERE,
                                   position=self.goal_position, scale=(0.05, 0.05, 0.05),
                                   color=(1.0, 0.0, 0.0, 0.8))
        self.goal_marker_pub.publish(marker)

        # Publish the same goal position to the /goal_position topic
        goal_msg = PoseStamped()
        goal_msg.header = marker.header
        goal_msg.pose.position.x, goal_msg.pose.position.y, goal_msg.pose.position.z = [float(x) for x in self.goal_position]
        self.goal_pub.publish(goal_msg)

    def publish_ee_marker(self, stamp, ee_x, ee_y, ee_z):
        marker = self._make_marker(stamp, "ee_offset", 2, Marker.SPHERE,
                                   position=(ee_x, ee_y, ee_z), scale=(0.03, 0.03, 0.03),
                                   color=(0.0, 0.0, 1.0, 0.8), lifetime=0.5)
        self.ee_marker_pub.publish(marker)

    def _publish_marker(self, stamp, px, py, pz, theta, is_lost=False, is_grasped=False):
        if is_grasped:
            color = (1.0, 0.5, 0.0, 0.8)  # Orange
        elif is_lost:
            color = (1.0, 0.0, 0.0, 0.6)  # Red
        else:
            color = (0.0, 1.0, 0.0, 0.6)  # Green
        marker = self._make_marker(stamp, "predicted_object", 0, Marker.SPHERE,
                                   position=(px, py, pz), scale=(0.05, 0.05, 0.05),
                                   color=color, lifetime=0.0)
        self.marker_pub.publish(marker)



    def publish_fitted_rectangle_marker(self, stamp, box_3d, pz):
        marker = self._make_marker(stamp, "fitted_rectangle", 3, Marker.LINE_STRIP,
                                   scale=(0.005, 0.0, 0.0), color=(1.0, 1.0, 0.0, 0.8), lifetime=0.5)
        for pt in box_3d:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pz)
            marker.points.append(p)
        p = Point()
        p.x, p.y, p.z = float(box_3d[0][0]), float(box_3d[0][1]), float(pz)
        marker.points.append(p)
        self.fitted_rect_pub.publish(marker)

    def _publish_prediction_helper(self, header):
        if self.last_predicted_pose is None:
            return
        px, py, pz = self.last_predicted_pose
        theta = self.last_predicted_yaw

        self._publish_pose_and_tf(header.stamp, px, py, pz, theta, is_lost=False, is_grasped=self.is_grasped)

        # Debug Logging
        self._dbg_counter += 1
        if self._dbg_counter % 10 == 0:
            self.get_logger().info(
                f"[PRED in {self.parent_frame}] pos=({px:.3f}, {py:.3f}, {pz:.3f}) "
                f"yaw={theta:.3f}"
            )



    def image_callback(self, msg):
        # 1. Convert ROS RGB image using CvBridge
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge RGB conversion error: {e}")
            return

        # Initialize debug saving variables
        rgb_mask = None
        final_mask = None
        coords = None
        box_3d = None

        # Check end-effector position for override & distance monitoring
        current_ee_pose = None
        try:
            trans_ee = self.tf_buffer.lookup_transform(
                self.parent_frame,
                'gripper_right_grasping_link',
                rclpy.time.Time()
            )
            ee_x = trans_ee.transform.translation.x
            ee_y = trans_ee.transform.translation.y
            ee_z = trans_ee.transform.translation.z
            
            qx = trans_ee.transform.rotation.x
            qy = trans_ee.transform.rotation.y
            qz = trans_ee.transform.rotation.z
            qw = trans_ee.transform.rotation.w

            # Convert quaternion to yaw
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            ee_yaw = math.atan2(siny_cosp, cosy_cosp)
            
            current_ee_pose = (ee_x, ee_y, ee_z, ee_yaw)
            ################################################
            # Publish EE marker for debugging in rviz
            self.publish_ee_marker(msg.header.stamp, ee_x, ee_y, ee_z)
            ##############################################
        except Exception as e:
            self.get_logger().warn(f"Failed to lookup EE transform: {e}", throttle_duration_sec=3.0)

        # If object is grasped, override object pose with EE pose directly
        if self.is_grasped:
            self._publish_grasped_ee_pose(msg.header, cv_img)
            return



        if self.intrinsics is None:
            self.get_logger().warn("Waiting for camera_info intrinsics...", throttle_duration_sec=3.0)
            return

        # Fetch latest depth image matching this color image (or as close as possible)
        depth_msg = None
        with self.latest_depth_lock:
            if self.latest_depth_msg is not None:
                depth_msg = self.latest_depth_msg

        if depth_msg is None:
            self.get_logger().warn("Waiting for depth images...", throttle_duration_sec=3.0)
            return

        # 2. Convert depth image using CvBridge
        try:
            if '32F' in depth_msg.encoding:
                cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                depth_m_full = cv_depth.copy()
            else:
                cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
                depth_m_full = cv_depth.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f"CvBridge depth conversion error: {e}")
            self.handle_detection_failure(msg.header, cv_img)
            return

        h, w, _ = cv_img.shape

        # 3. Object Detection
        bbox, conf = self.detect_object_yolo(cv_img)
        detection_label = f"YOLO (conf: {conf:.2f})"

        if bbox is None:
            self.handle_detection_failure(msg.header, cv_img)
            return

        x1, y1, x2, y2 = bbox

        # Bounding box bounds check (color-image space)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            self.handle_detection_failure(msg.header, cv_img)
            return

        camera_frame = self.camera_frame if self.camera_frame else msg.header.frame_id
        try:
            trans = self.tf_buffer.lookup_transform(
                self.parent_frame,
                camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            T_robot_camera = self.transform_to_matrix(trans)
        except Exception as e:
            self.get_logger().warn(f"TF Transform failed: {e}", throttle_duration_sec=3.0)
            self.handle_detection_failure(msg.header, cv_img)
            return

        # Depth success flag initialization
        fx, fy, cx, cy = self.intrinsics
        depth_success = False
        box_3d_val = None
        px, py, pz = 0.0, 0.0, 0.0

        # 4. Color preprocessing on RGB ROI (convert to HSV space and segment)
        rgb_mask = None
        try:
            roi = cv_img[y1:y2, x1:x2]
            if roi.size > 0:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                rh, rw = roi.shape[:2]
                
                cy1, cy2 = int(rh * 0.20), int(rh * 0.80)
                cx1, cx2 = int(rw * 0.20), int(rw * 0.80)
                center_pixels = roi_hsv[cy1:cy2, cx1:cx2]
                
                if center_pixels.size > 0:
                    hues_rad = center_pixels[:, :, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
                    sin_mean = np.mean(np.sin(hues_rad))
                    cos_mean = np.mean(np.cos(hues_rad))
                    cand_h_rad = np.arctan2(sin_mean, cos_mean)
                    if cand_h_rad < 0:
                        cand_h_rad += 2.0 * np.pi
                    cand_h = (cand_h_rad * (180.0 / np.pi) / 2.0) % 180.0
                    cand_s = float(np.median(center_pixels[:, :, 1]))
                    cand_v = float(np.median(center_pixels[:, :, 2]))
                    dominant_h = cand_h
                    dominant_s = cand_s
                    dominant_v = cand_v
                    
                    diff_h = np.abs(roi_hsv[:, :, 0].astype(np.int32) - dominant_h)
                    diff_h = np.minimum(diff_h, 180 - diff_h)
                    
                    diff_s = np.abs(roi_hsv[:, :, 1].astype(np.int32) - dominant_s)
                    diff_v = np.abs(roi_hsv[:, :, 2].astype(np.int32) - dominant_v)
                    
                    h_mask = diff_h < self.hsv_h_thresh
                    s_mask = (diff_s < self.hsv_s_thresh) & (roi_hsv[:, :, 1] > 20)
                    v_mask = (diff_v < self.hsv_v_thresh) & (roi_hsv[:, :, 2] > 20)
                    
                    rgb_mask = (h_mask & s_mask & v_mask).astype(np.uint8) * 255
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)
                    rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)
        except Exception as e:
            self.get_logger().warn(f"HSV color segmentation failed: {e}")

        # Depth mapping and 3D Back-projection logic
        if rgb_mask is not None and np.sum(rgb_mask > 0) >= 5:
            dh, dw = depth_m_full.shape[:2]
            scale_x = dw / w
            scale_y = dh / h

            y_indices, x_indices = np.where(rgb_mask > 0)
            u_coords = x_indices + x1
            v_coords = y_indices + y1

            du_coords = (u_coords * scale_x).astype(np.int32)
            dv_coords = (v_coords * scale_y).astype(np.int32)
            du_coords = np.clip(du_coords, 0, dw - 1)
            dv_coords = np.clip(dv_coords, 0, dh - 1)

            depths = depth_m_full[dv_coords, du_coords]

            valid_mask = (depths > 0.01) & (depths < self.max_depth_m) & np.isfinite(depths)
            
            if np.sum(valid_mask) >= 5:
                inlier_depths = depths[valid_mask]
                median_depth = np.median(inlier_depths)
                inlier_mask = np.abs(depths - median_depth) < 0.0025
                final_mask = valid_mask & inlier_mask



                if np.sum(final_mask) >= 5:
                    inlier_u = u_coords[final_mask]
                    inlier_v = v_coords[final_mask]
                    inlier_z = depths[final_mask]

                    # 6. Back-project to 3D in camera optical frame
                    X_cam = (inlier_u - cx) * inlier_z / fx
                    Y_cam = (inlier_v - cy) * inlier_z / fy
                    Z_cam = inlier_z

                    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)

                    # 7. Transform to base_footprint frame
                    points_cam_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
                    points_robot = (T_robot_camera @ points_cam_h.T).T[:, :3]



                    # 8. Fit 2D Oriented Bounding Box directly on the horizontal 3D coordinates
                    try:
                        coords = points_robot[:, :2].astype(np.float32)
                        rect_3d = cv2.minAreaRect(coords)
                        box_3d = cv2.boxPoints(rect_3d)
                        box_3d_val = box_3d
                        px = float(rect_3d[0][0])
                        py = float(rect_3d[0][1])
                        

                        pz = float((points_robot[:, 2].max() + points_robot[:, 2].min()) / 2.0)
                        
                        p0, p1, p2, p3 = box_3d
                        v1 = p1 - p0
                        v2 = p2 - p1
                        angle1 = math.atan2(v1[1], v1[0])
                        angle2 = math.atan2(v2[1], v2[0])
                        
                        wrapped_angle1 = wrap_to_pi_over_2(angle1)
                        wrapped_angle2 = wrap_to_pi_over_2(angle2)
                        
                        # Select nearest edge to keep yaw within [-pi/4, pi/4]
                        if abs(wrapped_angle1) < abs(wrapped_angle2):
                            theta = wrapped_angle1
                        else:
                            theta = wrapped_angle2
                            
                        depth_success = True
                    except Exception as e:
                        self.get_logger().warn(f"3D horizontal minAreaRect pose fitting failed: {e}")

        # Check success and apply Fallback logic
        if not depth_success:
            self.get_logger().warn("RGB-D depth pipeline failed.", throttle_duration_sec=3.0)
            self.handle_detection_failure(msg.header, cv_img)
            return



        # Apply constant target parent frame offset
        px += self.pose_offset[0]
        py += self.pose_offset[1]
        pz += self.pose_offset[2]



        if self.ema_x is None:
            self.ema_x = ExponentialMovingAverage(px, alpha=EMA_POS_ALPHA_DEFAULT)
            self.ema_y = ExponentialMovingAverage(py, alpha=EMA_POS_ALPHA_DEFAULT)
            self.ema_z = ExponentialMovingAverage(pz, alpha=EMA_POS_ALPHA_DEFAULT)
            self.ema_yaw = ExponentialMovingAverageYaw(theta, alpha=EMA_YAW_ALPHA_DEFAULT)
            self.yaw_locked = False
        else:
            # Set position alpha dynamically based on distance to EE
            is_near_ee = False
            if current_ee_pose is not None:
                ee_x, ee_y, ee_z, _ = current_ee_pose
                dist_to_ee = math.sqrt((px - ee_x)**2 + (py - ee_y)**2 + (pz - ee_z)**2)
                is_near_ee = (dist_to_ee < EE_NEAR_THRESHOLD_M)

            alpha_pos = EMA_POS_ALPHA_NEAR_EE if is_near_ee else EMA_POS_ALPHA_DEFAULT
            self.ema_x.alpha = alpha_pos
            self.ema_y.alpha = alpha_pos
            self.ema_z.alpha = alpha_pos

            # --- Yaw Lock Logic (3-second median calibration, permanent lock) ---
            import time as _time
            now = _time.monotonic()

            if not self.yaw_locked:
                # Initialise calibration window on first update
                if self.yaw_calib_start is None:
                    self.yaw_calib_start = now

                # Accumulate raw predictions, wrapping to [-pi/4, pi/4] symmetry window
                # so the median is meaningful for a 90-deg-symmetric object
                self.yaw_calib_buffer.append(theta)

                if now - self.yaw_calib_start >= 3:
                    # 1 second elapsed — compute median and lock permanently
                    locked_yaw = float(np.median(self.yaw_calib_buffer))
                    self.ema_yaw = ExponentialMovingAverageYaw(locked_yaw, alpha=EMA_YAW_ALPHA_LOCKED)
                    self.yaw_locked = True
                    self.get_logger().info(
                        f"[YAW LOCKED] median from {len(self.yaw_calib_buffer)} samples = "
                        f"{math.degrees(locked_yaw):.1f} deg ({locked_yaw:.4f} rad)"
                    )


            self.ema_x.update(px)
            self.ema_y.update(py)
            self.ema_z.update(pz)
            # ema_yaw is only updated during calibration (alpha=0.0 after lock freezes it)

        px = float(self.ema_x.x[0])
        py = float(self.ema_y.x[0])
        pz = float(self.ema_z.x[0])
        theta = float(self.ema_yaw.get_yaw())

        # Fixed yaw override
        if self.fixed_yaw_rad is not None:
            theta = self.fixed_yaw_rad

        self.last_predicted_pose = (px, py, pz)
        self.last_predicted_yaw = theta

        # Publish prediction
        if box_3d_val is not None:
            box_3d_val = box_3d_val.copy()
            box_3d_val[:, 0] += self.pose_offset[0]
            box_3d_val[:, 1] += self.pose_offset[1]
            self.publish_fitted_rectangle_marker(msg.header.stamp, box_3d_val, pz)

        self._publish_prediction_helper(msg.header)

        # Queue debug saving & visualization task asynchronously
        self.queue_visualization_task(
            header=msg.header,
            cv_img=cv_img,
            x1=x1, y1=y1, x2=x2, y2=y2,
            rgb_mask=rgb_mask,
            final_mask=final_mask,
            u_coords=u_coords,
            v_coords=v_coords,
            coords=coords,
            box_3d=box_3d,
            px=px, py=py, pz=pz, theta=theta,
            box_3d_val=box_3d_val,
            is_lost=False,
            status_text=detection_label
        )

    def queue_visualization_task(self, header, cv_img, x1=None, y1=None, x2=None, y2=None,
                                 rgb_mask=None, final_mask=None, u_coords=None, v_coords=None,
                                 coords=None, box_3d=None, px=None, py=None, pz=None, theta=None,
                                 box_3d_val=None, is_lost=False, is_grasped=False, status_text="LOST"):
        if cv_img is None:
            return

        projected_center = None
        projected_box = None
        
        # Calculate inverse TF transform (parent_frame to camera_frame) to project to camera image
        if px is not None and py is not None and pz is not None:
            try:
                camera_frame = self.camera_frame if self.camera_frame else header.frame_id
                trans_inv = self.tf_buffer.lookup_transform(
                    camera_frame,
                    self.parent_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                T_camera_robot = self.transform_to_matrix(trans_inv)
                
                # 1. Project Center point
                P_robot_center = np.array([px, py, pz, 1.0])
                P_cam_center = T_camera_robot @ P_robot_center
                if P_cam_center[2] > 0.1 and self.intrinsics is not None:
                    cfx, cfy, ccx, ccy = self.intrinsics
                    u_c = int((P_cam_center[0] * cfx) / P_cam_center[2] + ccx)
                    v_c = int((P_cam_center[1] * cfy) / P_cam_center[2] + ccy)
                    projected_center = (u_c, v_c)
                    
                # 2. Project 3D OBB points
                if box_3d_val is not None and self.intrinsics is not None:
                    cfx, cfy, ccx, ccy = self.intrinsics
                    projected_box = []
                    for pt in box_3d_val:
                        P_robot_pt = np.array([pt[0], pt[1], pz, 1.0])
                        P_cam_pt = T_camera_robot @ P_robot_pt
                        if P_cam_pt[2] > 0.1:
                            u_pt = int((P_cam_pt[0] * cfx) / P_cam_pt[2] + ccx)
                            v_pt = int((P_cam_pt[1] * cfy) / P_cam_pt[2] + ccy)
                            projected_box.append((u_pt, v_pt))
            except Exception as e:
                self.get_logger().warn(f"Failed to compute projection: {e}", throttle_duration_sec=3.0)

        # Prepare copies of arrays to put in the queue
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() - int(time.time())) * 1000)
        ts_str = f"{ts}_{ms:03d}"

        cv_img_copy = cv_img.copy()
        
        if self.debug_save_dir is not None:
            roi_bgr_copy = None
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                if x2 > x1 and y2 > y1:
                    roi_bgr_copy = cv_img[y1:y2, x1:x2].copy()

            hsv_mask_colored_copy = None
            if roi_bgr_copy is not None and rgb_mask is not None:
                hsv_mask_colored_copy = cv2.bitwise_and(roi_bgr_copy, roi_bgr_copy, mask=rgb_mask.copy())

            depth_inlier_canvas_copy = None
            if final_mask is not None and u_coords is not None and v_coords is not None and x1 is not None and y1 is not None:
                depth_inlier_canvas_copy = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                surviving_u = u_coords[final_mask] - x1
                surviving_v = v_coords[final_mask] - y1
                valid_roi = (
                    (surviving_u >= 0) & (surviving_u < (x2 - x1)) &
                    (surviving_v >= 0) & (surviving_v < (y2 - y1))
                )
                depth_inlier_canvas_copy[surviving_v[valid_roi], surviving_u[valid_roi]] = 255

            coords_copy = coords.copy() if coords is not None else None
            box_3d_copy = box_3d.copy() if box_3d is not None else None
        else:
            roi_bgr_copy = hsv_mask_colored_copy = depth_inlier_canvas_copy = None
            coords_copy = box_3d_copy = None

        proj_box_np = None
        if projected_box is not None and len(projected_box) >= 4:
            proj_box_np = np.array(projected_box[:4], dtype=np.int32)

        try:
            self.save_queue.put_nowait({
                'header': header,
                'ts': ts_str,
                'cv_img': cv_img_copy,
                'roi_bgr': roi_bgr_copy,
                'hsv_mask_colored': hsv_mask_colored_copy,
                'depth_inlier_canvas': depth_inlier_canvas_copy,
                'coords': coords_copy,
                'box_3d': box_3d_copy,
                'projected_center': projected_center,
                'projected_box': proj_box_np,
                'bbox': (x1, y1, x2, y2),
                'is_lost': is_lost,
                'is_grasped': is_grasped,
                'status_text': status_text,
                'px': px,
                'py': py,
                'pz': pz,
                'theta': theta
            })
        except queue.Full:
            self.get_logger().warn("Visualizer/save queue is full, dropping debug snapshots.", throttle_duration_sec=3.0)

    def _save_worker(self):
        while rclpy.ok():
            try:
                task = self.save_queue.get(timeout=1.0)
            except Exception:
                continue
            
            try:
                header = task['header']
                ts = task['ts']
                cv_img = task['cv_img']
                roi_bgr = task['roi_bgr']
                hsv_mask_colored = task['hsv_mask_colored']
                depth_inlier_canvas = task['depth_inlier_canvas']
                coords = task['coords']
                box_3d = task['box_3d']
                projected_center = task['projected_center']
                projected_box = task['projected_box']
                bbox = task.get('bbox', (None, None, None, None))
                is_lost = task['is_lost']
                is_grasped = task.get('is_grasped', False)
                status_text = task['status_text']
                px = task['px']
                py = task['py']
                pz = task['pz']
                theta = task['theta']

                # --- 1. Draw annotation on cv_img ---
                # 1. Pos on top
                if px is not None and py is not None and pz is not None:
                    cv2.putText(cv_img, f"Pos: ({px:.3f}, {py:.3f}, {pz:.3f}) m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 2. Yaw in the middle
                if theta is not None:
                    cv2.putText(cv_img, f"Yaw: {math.degrees(theta):.1f} deg", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 3. Status text & flag banner
                y_offset = 85 if (px is not None or theta is not None) else 35
                if is_grasped:
                    text_color = (0, 140, 255)  # Orange (BGR)
                    h_img, w_img = cv_img.shape[:2]
                    cv2.rectangle(cv_img, (w_img - 180, 15), (w_img - 10, 50), (0, 140, 255), -1)
                    cv2.putText(cv_img, "GRASPED", (w_img - 165, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                elif is_lost:
                    text_color = (0, 0, 255)  # Red (BGR)
                else:
                    text_color = (0, 255, 0)  # Green (BGR)

                cv2.putText(cv_img, f"Status: {status_text}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                
                if self.show_obb and projected_center is not None:
                    cv2.circle(cv_img, projected_center, 5, (0, 0, 255), -1)

                if self.show_obb and projected_box is not None:
                    pts = projected_box.reshape((-1, 1, 2))
                    cv2.polylines(cv_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

                # --- 2. Publish annotated image ---
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                    vis_msg.header = header
                    self.vis_pub.publish(vis_msg)
                except Exception as e:
                    self.get_logger().error(f"Failed to publish annotated image in background: {e}")

                # --- 3. Save files to disk (if debug_save_dir is enabled) ---
                if self.debug_save_dir is not None:
                    x1_b, y1_b, x2_b, y2_b = bbox
                    if x1_b is not None and y1_b is not None and x2_b is not None and y2_b is not None:
                        cv2.rectangle(cv_img, (int(x1_b), int(y1_b)), (int(x2_b), int(y2_b)), (0, 255, 0), 1)

                    cv2.imwrite(os.path.join(self.debug_save_dir, f"{ts}_camera_view.png"), cv_img)
                    
                    if roi_bgr is not None:
                        cv2.imwrite(os.path.join(self.debug_save_dir, f"{ts}_roi.png"), roi_bgr)
                    
                    if hsv_mask_colored is not None:
                        cv2.imwrite(os.path.join(self.debug_save_dir, f"{ts}_hsv_mask.png"), hsv_mask_colored)
                    
                    if depth_inlier_canvas is not None:
                        cv2.imwrite(os.path.join(self.debug_save_dir, f"{ts}_depth_inlier_mask.png"), depth_inlier_canvas)
                    
                    if coords is not None and box_3d is not None:
                        self._save_pointcloud_rect_plot_helper(coords, box_3d, os.path.join(self.debug_save_dir, f"{ts}_xy_fit.png"))
                    
                    # Stop saving after the first sequence is written to disk
                    self.get_logger().info(f"[DEBUG SAVE] Saved one sequence of samples to {self.debug_save_dir}. Disabling further saving.")
                    self.debug_save_dir = None

            except Exception as e:
                self.get_logger().error(f"Error in background save/visualize worker: {e}")
            finally:
                self.save_queue.task_done()

    def _save_pointcloud_rect_plot_helper(self, coords, box_3d, filename):
        try:
            # Create a black image
            img_w, img_h = 600, 600
            canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            
            # Find min and max coordinates to scale and center
            all_pts = np.vstack([coords, box_3d])
            x_min, y_min = np.min(all_pts, axis=0)
            x_max, y_max = np.max(all_pts, axis=0)
            
            # Add padding
            dx = x_max - x_min
            dy = y_max - y_min
            padding = max(dx, dy, 0.01) * 0.15
            
            x_min -= padding
            x_max += padding
            y_min -= padding
            y_max += padding
            
            range_x = x_max - x_min
            range_y = y_max - y_min
            
            # Scale to fit canvas, keeping aspect ratio
            scale = min((img_w - 60) / range_x, (img_h - 60) / range_y)
            
            def to_pixels(pt):
                # Center the plot on the canvas
                px = int(30 + (pt[0] - x_min) * scale)
                py = int(30 + (pt[1] - y_min) * scale)
                return px, py
            
            # Draw origin grid / coordinate axes
            if x_min <= 0 <= x_max and y_min <= 0 <= y_max:
                ox, oy = to_pixels((0.0, 0.0))
                cv2.line(canvas, (0, oy), (img_w, oy), (80, 80, 80), 1)
                cv2.line(canvas, (ox, 0), (ox, img_h), (80, 80, 80), 1)
                
            # Draw point cloud (Blue dots)
            for pt in coords:
                px, py = to_pixels(pt)
                cv2.circle(canvas, (px, py), 2, (255, 100, 100), -1)
                
            # Draw fitted oriented rectangle (Yellow outline)
            rect_pts = []
            for pt in box_3d:
                px, py = to_pixels(pt)
                rect_pts.append([px, py])
                
            rect_pts = np.array(rect_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [rect_pts], isClosed=True, color=(0, 255, 255), thickness=2)
            
            cv2.imwrite(filename, canvas)
        except Exception as e:
            self.get_logger().error(f"Failed to save pointcloud plot to {filename}: {e}")





def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="YOLO + HSV + Geometric 3D Pose Estimator")

    parser.add_argument(
        "--yolo_model",
        type=str,
        default="tiago_single_class_yolo26/weights/best.pt",
        help="Path to YOLO model weight file (.pt)"
    )
    parser.add_argument(    
        "--yolo_conf",
        type=float,
        default=0.45,
        help="Confidence threshold for YOLO detection."
    )
    parser.add_argument(
        "--parent_frame",
        type=str,
        default="base_footprint",
        help="Parent target coordinate frame (default: base_footprint)."
    )
    parser.add_argument(
        "--depth_topic",
        type=str,
        default="/head_front_camera/aligned_depth_to_color/image_raw",
        help="Topic for aligned depth image (default: /head_front_camera/aligned_depth_to_color/image_raw)."
    )
    parser.add_argument(
        "--pose_offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Constant 3-axis offset (X, Y, Z) in meters in the target parent frame (default: [0.0, 0.0, 0.0])."
    )
    parser.add_argument(
        "--goal_position",
        type=float,
        nargs=3,
        default=[0.72, -0.716, 0.60],
        help="3D goal position [X, Y, Z] in meters to publish as a marker (default: [0.72, -0.716, 0.60])."
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=2.5,
        help="Maximum valid depth in metres for the depth filter (default: 2.5)."
    )
    parser.add_argument(
        "--hsv_thresh",
        type=int,
        nargs=3,
        default=[100, 100, 80],
        help="HSV segmentation thresholds [H, S, V] (default: [100, 100, 80])."
    )
    parser.add_argument(
        "--debug_save_dir",
        type=str,
        default=None,
        help="If set, save debug snapshots (ROI, HSV mask, depth-inlier mask) to this directory on every detection."
    )
    parser.add_argument(
        "--show_obb",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable/disable drawing the oriented bounding box on camera visualization topic (default: True, accepts True/False)."
    )
    parser.add_argument(
        "--no-show_obb", "--no_show_obb",
        dest="show_obb",
        action="store_false",
        help="Disable drawing the oriented bounding box on camera visualization topic."
    )
    parser.add_argument(
        "--depth_gazebo",
        action="store_true",
        help="Use Gazebo depth topic (/head_front_camera/depth/image_rect_raw)."
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Fix the published yaw to this constant value in degrees (default: None = detect from image)."
    )

    args, unknown = parser.parse_known_args()

    if args.depth_gazebo:
        args.depth_topic = "/head_front_camera/depth/image_rect_raw"

    rclpy.init()
    node = YoloDepthPoseNode(
        yolo_model_path=args.yolo_model,
        yolo_conf=args.yolo_conf,
        parent_frame=args.parent_frame,
        depth_topic=args.depth_topic,
        pose_offset=args.pose_offset,
        goal_position=args.goal_position,
        max_depth_m=args.max_depth,
        hsv_thresh=args.hsv_thresh,
        debug_save_dir=args.debug_save_dir,
        fixed_yaw_deg=args.yaw,
        show_obb=args.show_obb
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()