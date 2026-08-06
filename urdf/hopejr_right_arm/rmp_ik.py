from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid

import omni.usd
from pxr import UsdLux, Sdf, UsdGeom

import numpy as np
import os

# ------------------------
# 경로 설정
# ------------------------
base_path = "C:/Users/user/Documents/hopejr_right_arm/src/hopejr_right_arm_description/urdf/hopejr_right_arm"
usd_path = os.path.join(base_path, "hopejr_right_arm.usd")
urdf_path = "C:/Users/user/Documents/hopejr_right_arm/src/hopejr_right_arm_description/urdf/hopejr_right_arm.urdf"
robot_description_path = os.path.join(base_path, "robot_description.yaml")
rmpflow_config_path = os.path.join(base_path, "rmpflow_config.yaml")

# ------------------------
# World 생성
# ------------------------
world = World(stage_units_in_meters=1.0)

add_reference_to_stage(usd_path, "/World/hopejr")

stage = omni.usd.get_context().get_stage()

# ------------------------
# Light 추가
# ------------------------
light_prim = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Light"))
light_prim.CreateIntensityAttr(5000)

# ------------------------
# world reset
# ------------------------
world.reset()

# ------------------------
# Robot articulation wrap
# ------------------------
robot = Articulation("/World/hopejr")
robot.initialize()

# ------------------------
# RMPFlow Solver 초기화
# ------------------------
rmp_solver = RmpFlow(
    robot_description_path=robot_description_path,
    urdf_path=urdf_path,
    rmpflow_config_path=rmpflow_config_path,
    end_effector_frame_name="hand_palm",
    maximum_substep_size=0.00333
)

# ArticulationMotionPolicy를 사용하여 로봇과 solver를 연결
motion_policy = ArticulationMotionPolicy(robot, rmp_solver)

# ------------------------
# 타겟 큐브 생성
# ------------------------
initial_cube_position = np.array([0.0, 0.3, 0.2])

target_cube = VisualCuboid(
    prim_path="/World/target",
    name="target",
    position=initial_cube_position,
    size=0.05,
    color=np.array([1.0, 0.0, 0.0])
)

target_prim = stage.GetPrimAtPath("/World/target")

# ------------------------
# Loop
# ------------------------
while simulation_app.is_running():

    world.step(render=True)

    # 1. 타겟 큐브 위치 읽기
    xform = UsdGeom.Xformable(target_prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    cube_position = np.array(world_transform.ExtractTranslation())

    # 2. 좌표계 변환 (사용자 커스텀 변환 유지)
    target_position = np.array([
        -cube_position[0],
         cube_position[1],
        -cube_position[2]
    ])

    # 3. RMPFlow 타겟 설정
    rmp_solver.set_end_effector_target(
        target_position=target_position,
        target_orientation=None # 필요시 np.array([1, 0, 0, 0]) 등 추가
    )

    # 4. 액션 계산 및 적용
    actions = motion_policy.get_next_articulation_action()
    robot.apply_action(actions)

    # 5. EE 실제 위치 확인 및 오차 출력
    ee_prim = stage.GetPrimAtPath("/World/hopejr/hand_hand")
    ee_xform = UsdGeom.Xformable(ee_prim)
    ee_world = ee_xform.ComputeLocalToWorldTransform(0)
    ee_position = np.array(ee_world.ExtractTranslation())

    # 좌표계 반전된 타겟과 비교
    error = np.linalg.norm(target_position - ee_position)
    if world.current_time_step_index % 60 == 0:
        print(f"Position error: {error:.4f}")

simulation_app.close()
