from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.motion_generation import ArticulationKinematicsSolver
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid

import omni.usd
from pxr import UsdLux, Sdf, UsdGeom

import numpy as np

# ------------------------
# World 생성
# ------------------------
world = World(stage_units_in_meters=1.0)

robot_root_path = "C:/IsaacsimAssets/robots/hopejr_right_arm/"

usd_path=robot_root_path+"hopejr_right_arm.usd"
add_reference_to_stage(usd_path, "/World/hopejr")

stage = omni.usd.get_context().get_stage()

# ------------------------
# Light 추가 (Isaac 5.1.0 안정 방식)
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
# IK Solver 초기화
# ------------------------
lula_solver = LulaKinematicsSolver(
    robot_description_path=robot_root_path+"robot_description.yaml",
    urdf_path=robot_root_path+"hopejr_right_arm.urdf"
)

print(f"Frames: {lula_solver.get_all_frame_names()}")

art_ik = ArticulationKinematicsSolver(
    robot,
    lula_solver,
    end_effector_frame_name="hand_palm"
)

# ------------------------
# 타겟 큐브 생성 (초기 위치)
# ------------------------
initial_cube_position = np.array([0.0, 0.3, 0.2])

target_cube = VisualCuboid(
    prim_path="/World/target",
    name="target",
    position=initial_cube_position,
    size=0.05,
)

target_prim = stage.GetPrimAtPath("/World/target")

# ------------------------
# Loop
# ------------------------
while simulation_app.is_running():

    world.step(render=True)

    # ------------------------
    # 1️⃣ 큐브 world 위치 읽기
    # ------------------------
    xform = UsdGeom.Xformable(target_prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    cube_position = np.array(world_transform.ExtractTranslation())

    # ------------------------
    # 2️⃣ 좌표계 변환 (x, z 반전)
    # ------------------------
    target_position = cube_position

    # ------------------------
    # 3️⃣ IK 계산
    # ------------------------
    actions, success = art_ik.compute_inverse_kinematics(
        target_position,
        None
    )

    # ------------------------
    # 4️⃣ EE 실제 위치 확인
    # ------------------------
    ee_prim = stage.GetPrimAtPath("/World/hopejr/hand_hand")
    ee_xform = UsdGeom.Xformable(ee_prim)
    ee_world = ee_xform.ComputeLocalToWorldTransform(0)
    ee_position = np.array(ee_world.ExtractTranslation())

    print("Position error:",
          np.linalg.norm(target_position - ee_position))

    # ------------------------
    # 5️⃣ Joint 적용
    # ------------------------
    if success:
        robot.apply_action(actions)

simulation_app.close()