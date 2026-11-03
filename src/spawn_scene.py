#!/usr/bin/env python3.10
"""
spawn_scene.py — Spawns a table and a cube into a running Gazebo simulation.

Run it from a sourced ROS 2 environment while Gazebo is running:
    python3 src/spawn_scene.py

All configurable parameters are at the top of the file.
"""

import math

# =============================================================================
# CONFIGURATION 
# =============================================================================

# --- Table ---
TABLE_POS_X = 0.75   # metres in front of the robot
TABLE_POS_Y = 0.00   # metres lateral offset (negative = right)
TABLE_POS_Z = 0.00   # metres vertical (leave 0 for ground level)
TABLE_YAW   = 0.00   # rotation around vertical axis (radians)

TABLE_SURFACE_WIDTH  = 0.70   # metres (X axis)
TABLE_SURFACE_DEPTH  = 0.70   # metres (Y axis)
TABLE_SURFACE_HEIGHT = 0.03   # metres (thickness of the tabletop)
TABLE_LEG_RADIUS     = 0.03   # metres (cylinder radius)
TABLE_HEIGHT         = 0.45   # metres (floor to top surface)
TABLE_MASS           = 20.0   # kg

TABLE_COLOR_R = 0.05
TABLE_COLOR_G = 0.05
TABLE_COLOR_B = 0.05
TABLE_COLOR_A = 1.0

# --- Cube ---
CUBE_SIZE_X = 0.04   # metres
CUBE_SIZE_Y = 0.04   # metres
CUBE_SIZE_Z = 0.05   # metres
CUBE_COLLISION_SCALE = 0.8  # scale factor to reduce collision box size relative to visual size
CUBE_MASS   = 0.1   # kg
CUBE_FRICTION_MU  = 1.0
CUBE_FRICTION_MU2 = 1.0

# Placed on top of the table (auto-computed from TABLE_HEIGHT + CUBE_SIZE_Z/2)
CUBE_OFFSET_X = 0.2   # metres relative to table centre (X)
CUBE_OFFSET_Y = 0.1   # metres relative to table centre (Y)
CUBE_YAW      = 0.5    # rotation around vertical axis (radians relative to robot yaw)

CUBE_COLOR_R = 1.0
CUBE_COLOR_G = 0.0
CUBE_COLOR_B = 0.0
CUBE_COLOR_A = 1.0
CUBE_EMISSIVE_STRENGTH = 0.1  # Adds self-illumination (0.0 to 1.0) to make the red look bright/vibrant

# =============================================================================
# SDF GENERATION
# =============================================================================

def _material(r, g, b, a=1.0, emissive_strength=0.0) -> str:
    # By bypassing the fixed Gazebo/Red script and using direct ambient/diffuse,
    # we get a more vibrant color. Adding an emissive term makes it glow/look brighter.
    er = r * emissive_strength
    eg = g * emissive_strength
    eb = b * emissive_strength
    return f"""
          <material>
            <ambient>{r} {g} {b} {a}</ambient>
            <diffuse>{r} {g} {b} {a}</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
            <emissive>{er:.3f} {eg:.3f} {eb:.3f} {a}</emissive>
          </material>"""


def build_table_sdf() -> str:
    """Returns a full SDF string for a simple four-legged table."""
    leg_height = TABLE_HEIGHT - TABLE_SURFACE_HEIGHT
    leg_z      = leg_height / 2.0          # centre of leg from ground
    top_z      = leg_height + TABLE_SURFACE_HEIGHT / 2.0  # centre of top

    hw = TABLE_SURFACE_WIDTH  / 2.0        # half-width
    hd = TABLE_SURFACE_DEPTH  / 2.0        # half-depth
    lr = TABLE_LEG_RADIUS

    # Four leg positions (corners)
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
          </geometry>{_material(TABLE_COLOR_R * 0.7, TABLE_COLOR_G * 0.7, TABLE_COLOR_B * 0.7)}
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
        </geometry>{_material(TABLE_COLOR_R, TABLE_COLOR_G, TABLE_COLOR_B)}
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


def build_cube_sdf() -> str:
    """Returns a full SDF string for a coloured cube."""
    ix = CUBE_MASS / 12.0 * (CUBE_SIZE_Y**2 + CUBE_SIZE_Z**2)
    iy = CUBE_MASS / 12.0 * (CUBE_SIZE_X**2 + CUBE_SIZE_Z**2)
    iz = CUBE_MASS / 12.0 * (CUBE_SIZE_X**2 + CUBE_SIZE_Y**2)

    cube_z = TABLE_HEIGHT + CUBE_SIZE_Z / 2.0
    cube_x = TABLE_POS_X + CUBE_OFFSET_X
    cube_y = TABLE_POS_Y + CUBE_OFFSET_Y

    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="cube">
    <pose>0 0 0 0 0 0</pose>
    <plugin name="gazebo_link_attacher" filename="libgazebo_link_attacher.so"/>
    <link name="link">
      <inertial>
        <mass>{CUBE_MASS}</mass>
        <inertia>
          <ixx>{ix:.6f}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{iy:.6f}</iyy>
          <iyz>0.0</iyz>
          <izz>{iz:.6f}</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>{CUBE_SIZE_X} {CUBE_SIZE_Y} {CUBE_SIZE_Z}</size>
          </box>
        </geometry>{_material(CUBE_COLOR_R, CUBE_COLOR_G, CUBE_COLOR_B, emissive_strength=CUBE_EMISSIVE_STRENGTH)}
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>{CUBE_SIZE_X * CUBE_COLLISION_SCALE:.6f} {CUBE_SIZE_Y * CUBE_COLLISION_SCALE:.6f} {CUBE_SIZE_Z:.6f}</size>
          </box>
        </geometry>
        <surface>
          <contact>
            <ode>
              <min_depth>0.002</min_depth>
              <max_vel>0.0</max_vel>
              <kp>100000000.0</kp>
              <kd>100.0</kd>
            </ode>
          </contact>
          <friction>
            <ode>
              <mu>{CUBE_FRICTION_MU}</mu>
              <mu2>{CUBE_FRICTION_MU2}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
  </model>
</sdf>
"""


# =============================================================================
# SPAWNING LOGIC
# =============================================================================

def spawn(entity_name: str, sdf_string: str, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
    """
    Call the Gazebo /spawn_entity service directly via rclpy.
    This bypasses spawn_entity.py (and its broken lxml dependency) entirely.
    """
    import rclpy
    from rclpy.node import Node
    from gazebo_msgs.srv import SpawnEntity
    from geometry_msgs.msg import Pose
    import math

    print(f"\n[spawn_scene] Spawning '{entity_name}' at ({x:.2f}, {y:.2f}, {z:.2f}) yaw={yaw:.2f}...")

    rclpy.init(args=None)
    node = Node(f"spawn_{entity_name}_node")

    client = node.create_client(SpawnEntity, "/spawn_entity")

    if not client.wait_for_service(timeout_sec=5.0):
        print(f"  ✗ /spawn_entity service not available. Is Gazebo running?")
        node.destroy_node()
        rclpy.shutdown()
        return False

    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    # Convert yaw to quaternion (roll=0, pitch=0)
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)

    req = SpawnEntity.Request()
    req.name = entity_name
    req.xml = sdf_string
    req.initial_pose = pose
    req.reference_frame = "world"

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

    node.destroy_node()
    rclpy.shutdown()

    if future.done() and future.result() is not None:
        result = future.result()
        if result.success:
            print(f"  ✓ '{entity_name}' spawned successfully.")
            return True
        else:
            print(f"  ✗ Gazebo refused to spawn '{entity_name}': {result.status_message}")
            return False
    else:
        print(f"  ✗ Service call timed out for '{entity_name}'.")
        return False


def get_robot_pose() -> tuple[float, float, float, float]:
    """Returns (x, y, z, yaw) of the robot in the Gazebo world."""
    import subprocess
    for model_name in ["tiago-pro", "tiago_pro", "tiago"]:
        try:
            out = subprocess.check_output(["gz", "model", "-m", model_name, "-p"], text=True, timeout=2.0)
            parts = out.strip().split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                yaw = float(parts[5])
                print(f"[spawn_scene] Detected robot '{model_name}' pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}")
                return x, y, z, yaw
        except Exception:
            continue
    print("[spawn_scene] Failed to detect robot pose. Using defaults (0,0,0,0).")
    return 0.0, 0.0, 0.0, 0.0

def main():
    print("=" * 60)
    print("  Tiago Pro Gazebo Scene Spawner (Auto-Aligning)")
    print("=" * 60)

    rx, ry, rz, ryaw = get_robot_pose()

    # Calculate Table absolute pose
    # Rotated around robot's yaw
    table_x = rx + TABLE_POS_X * math.cos(ryaw) - TABLE_POS_Y * math.sin(ryaw)
    table_y = ry + TABLE_POS_X * math.sin(ryaw) + TABLE_POS_Y * math.cos(ryaw)
    table_z = rz + TABLE_POS_Z
    table_yaw = ryaw + TABLE_YAW

    # Calculate Cube absolute pose
    cube_dx = TABLE_POS_X + CUBE_OFFSET_X
    cube_dy = TABLE_POS_Y + CUBE_OFFSET_Y
    cube_dz = TABLE_HEIGHT + CUBE_SIZE_Z / 2.0

    cube_x = rx + cube_dx * math.cos(ryaw) - cube_dy * math.sin(ryaw)
    cube_y = ry + cube_dx * math.sin(ryaw) + cube_dy * math.cos(ryaw)
    cube_z = rz + cube_dz
    cube_yaw = ryaw + CUBE_YAW

    ok_table = spawn(
        entity_name="table",
        sdf_string=build_table_sdf(),
        x=table_x,
        y=table_y,
        z=table_z,
        yaw=table_yaw,
    )

    ok_cube = spawn(
        entity_name="cube",
        sdf_string=build_cube_sdf(),
        x=cube_x,
        y=cube_y,
        z=cube_z,
        yaw=cube_yaw,
    )

    print()
    if ok_table: #and ok_cube:
        print("✓ Scene ready! Table and cube spawned and aligned with robot.")
        print(f"  Robot pose           : ({rx:.3f}, {ry:.3f}, {rz:.3f}) yaw={ryaw:.3f}")
        print(f"  Table position       : ({table_x:.3f}, {table_y:.3f}, {table_z:.3f}) yaw={table_yaw:.3f}")
        print(f"  Cube position        : ({cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}) yaw={cube_yaw:.3f}")
    else:
        print("✗ One or more entities failed to spawn. Check Gazebo is running.")


if __name__ == "__main__":
    main()
