#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class GraspedCurrentPublisher(Node):
    def __init__(self):
        super().__init__('grasped_current_publisher')

        # Hysteresis thresholds (Amperes)
        self.activation_threshold   = 1.0    # A
        self.hysteresis             = 0.15   # A
        self.deactivation_threshold = self.activation_threshold - self.hysteresis  # 0.85 A

        # State variables
        self.current_a  = 0.0
        self.grasped    = False
        self.gripper_pos = 0.045   # m, default open position

        # Publisher — TRANSIENT_LOCAL (latching) so late subscribers always
        # receive the last known state immediately on connect.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub = self.create_publisher(Bool, '/grasped_current', latched_qos)

        # /diagnostics runs at ~87 Hz on PAL robots and carries motor current
        self.diag_sub = self.create_subscription(
            DiagnosticArray,
            '/diagnostics',
            self.diagnostics_callback,
            10
        )

        # /joint_states for gripper position only (effort is NaN for gripper)
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.get_logger().info(
            f"Grasped Current Publisher Node Initialized.\n"
            f"  Source             : /diagnostics (~87 Hz)\n"
            f"  Topic              : {self.pub.topic_name}\n"
            f"  Activation Thresh  : {self.activation_threshold} A\n"
            f"  Deactivation Thresh: {self.deactivation_threshold} A\n"
        )

    def joint_callback(self, msg: JointState):
        try:
            if 'gripper_right_finger_joint' in msg.name:
                idx = msg.name.index('gripper_right_finger_joint')
                self.gripper_pos = msg.position[idx]
        except Exception:
            pass

    def diagnostics_callback(self, msg: DiagnosticArray):
        found_actuator = False
        for status in msg.status:
            if 'gripper_right_finger_actuator' in status.name:
                for val in status.values:
                    if val.key == 'Current state':
                        try:
                            self.current_a = float(val.value.split()[0])
                            found_actuator = True
                        except (ValueError, IndexError):
                            pass

        if found_actuator:
            # Hysteresis logic
            prev_grasped = self.grasped
            if self.grasped:
                if self.current_a < self.deactivation_threshold:
                    self.grasped = False
            else:
                if self.current_a > self.activation_threshold:
                    self.grasped = True

            # Safety: gripper almost fully closed (<0.015 m) → no object
            if self.grasped and self.gripper_pos < 0.015:
                self.grasped = False

            if self.grasped != prev_grasped:
                self.get_logger().info(
                    f"Grasped status changed: {prev_grasped} -> {self.grasped} "
                    f"(current: {self.current_a:.3f} A, gripper: {self.gripper_pos:.4f} m)"
                )

        # Always publish last known state — even when the actuator entry is
        # missing from this diagnostics message — so the subscriber never
        # misses a grasped=True transition at 87 Hz.
        msg_pub = Bool()
        msg_pub.data = self.grasped
        self.pub.publish(msg_pub)

        if sys.stdout.isatty():
            status_str = "GRASPED" if self.grasped else "RELEASED"
            print(
                f"\rCurrent: {self.current_a:.3f} A | "
                f"Gripper: {self.gripper_pos:.4f} m | Status: {status_str}",
                end='', flush=True
            )


def main(args=None):
    rclpy.init(args=args)
    node = GraspedCurrentPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
