#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
import cv2
import numpy as np
from PIL import Image
import logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gen_world_from_image_gz")

# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
PALETTE_ORDER = ["avenue", "park", "water", "main_street", "side_street", "building"]
PALETTE_RGB = {
    "avenue":       (252, 203,  72),
    "park":         ( 21,  72,  26),
    "water":        ( 33,  45, 100),
    "main_street":  (240, 240, 241),
    "side_street":  ( 22,  20,  22),
    "building":     ( 12,  10,  13),
}
IDX = {name: i for i, name in enumerate(PALETTE_ORDER)}


def segment_by_palette(img_bgr: np.ndarray):
    # Use int32 to avoid overflow in squared distances (fixes NaN in sqrt)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int32)
    colors = np.array([PALETTE_RGB[name] for name in PALETTE_ORDER], dtype=np.int32)

    diff = img_rgb[:, :, None, :] - colors[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=3, dtype=np.int32)  # (H, W, K), non-negative

    label_map = np.argmin(dist2, axis=2).astype(np.uint8)  # (H, W)
    min_dist = np.sqrt(np.min(dist2, axis=2).astype(np.float32))  # (H, W), finite

    return label_map, min_dist


def build_masks(label_map, min_dist, building_tol=12.0, road_tol=18.0):
    building_idx = IDX["building"]
    road_indices = [IDX["avenue"], IDX["main_street"], IDX["side_street"]]

    mask_building = ((label_map == building_idx) & (min_dist <= building_tol)).astype(np.uint8) * 255
    mask_roads = (np.isin(label_map, road_indices) & (min_dist <= road_tol)).astype(np.uint8) * 255

    return mask_building, mask_roads


def compute_integral_image(mask_binary: np.ndarray) -> np.ndarray:
    return cv2.integral(mask_binary)


def rectangle_sum(integral_img, x: int, y: int, w: int, h: int) -> int:
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return int(integral_img[y2, x2] - integral_img[y1, x2] - integral_img[y2, x1] + integral_img[y1, x1])


def extract_building_boxes(
    building_mask_255: np.ndarray,
    roads_mask_255: np.ndarray,
    min_area: int = 800,
    min_side: int = 20,
    max_side: int = 300,
    max_aspect_ratio: float = 3.5,
    max_road_fraction: float = 0.02,
) -> list:
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(building_mask_255, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roads_binary = (roads_mask_255 > 0).astype(np.uint8)
    roads_integral = compute_integral_image(roads_binary)

    valid_buildings = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)

        if area < min_area:
            continue
        if w < min_side or h < min_side or w > max_side or h > max_side:
            continue

        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > max_aspect_ratio:
            continue

        road_pixels = rectangle_sum(roads_integral, x, y, w, h)
        road_fraction = road_pixels / float(w * h)
        if road_fraction > max_road_fraction:
            continue

        valid_buildings.append((area, x, y, w, h))

    return valid_buildings


# -----------------------------------------------------------------------------
# Materials (OGRE)
# -----------------------------------------------------------------------------
def write_ogre_material(out_dir: str, texture_filename: str = "finalmap.png") -> None:
    scripts_dir = os.path.join(out_dir, "materials", "scripts")
    textures_dir = os.path.join(out_dir, "materials", "textures")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(textures_dir, exist_ok=True)

    material_path = os.path.join(scripts_dir, "utm.material")
    material_txt = f"""material UTM/FinalMap
{{
  technique
  {{
    pass
    {{
      lighting on
      ambient 1 1 1 1
      diffuse 1 1 1 1
      specular 0 0 0 1

      texture_unit
      {{
        texture {texture_filename}
        scale 1 1
      }}
    }}
  }}
}}
"""
    with open(material_path, "w", encoding="utf-8") as f:
        f.write(material_txt)

    logger.info(f"Created OGRE material script: {material_path}")


# -----------------------------------------------------------------------------
# SDF generator
# -----------------------------------------------------------------------------
def make_world_sdf(
    W_px,
    H_px,
    boxes,
    resolution_m_per_px=0.2,
    random_height=True,
    height_fixed=20.0,
    seed=42,
):
    random.seed(seed)

    width_m = W_px * resolution_m_per_px
    height_m = H_px * resolution_m_per_px

    hm_uri = "file://heightmap.png"

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<sdf version="1.7">')
    lines.append('  <world name="utm_world">')
    lines.append('    <gravity>0 0 -9.81</gravity>')
    lines.append('    <physics name="default_physics" type="ode">')
    lines.append('      <max_step_size>0.002</max_step_size>')
    lines.append('      <real_time_factor>1.0</real_time_factor>')
    lines.append('      <real_time_update_rate>500</real_time_update_rate>')
    lines.append('    </physics>')

    lines.append('    <model name="map_ground">')
    lines.append('      <static>true</static>')
    lines.append('      <pose>0 0 0 0 0 0</pose>')
    lines.append('      <link name="link">')

    lines.append('        <collision name="collision">')
    lines.append('          <geometry>')
    lines.append('            <heightmap>')
    lines.append(f'              <uri>{hm_uri}</uri>')
    lines.append(f'              <size>{width_m:.6f} {height_m:.6f} 1.0</size>')
    lines.append('              <pos>0 0 0</pos>')
    lines.append('            </heightmap>')
    lines.append('          </geometry>')
    lines.append('        </collision>')

    lines.append('        <visual name="visual">')
    lines.append('          <pose>0 0 0.001 0 0 0</pose>')
    lines.append('          <geometry>')
    lines.append('            <plane>')
    lines.append('              <normal>0 0 1</normal>')
    lines.append(f'              <size>{width_m:.6f} {height_m:.6f}</size>')
    lines.append('            </plane>')
    lines.append('          </geometry>')
    lines.append('          <material>')
    lines.append('            <script>')
    lines.append('              <uri>file://materials/scripts</uri>')
    lines.append('              <uri>file://materials/textures</uri>')
    lines.append('              <name>UTM/FinalMap</name>')
    lines.append('            </script>')
    lines.append('          </material>')
    lines.append('        </visual>')

    lines.append('      </link>')
    lines.append('    </model>')

    lines.append('    <!-- Buildings generated from dark rectangles -->')
    for i, (area, x, y, w, h) in enumerate(boxes):
        cx = x + w / 2.0
        cy = y + h / 2.0

        wx = (cx - W_px / 2.0) * resolution_m_per_px
        wy = (H_px / 2.0 - cy) * resolution_m_per_px

        sx = w * resolution_m_per_px
        sy = h * resolution_m_per_px

        hz = random.uniform(8.0, 35.0) if random_height else float(height_fixed)
        wz = hz / 2.0

        lines.append(f'    <model name="building_{i:04d}">')
        lines.append('      <static>true</static>')
        lines.append(f'      <pose>{wx:.3f} {wy:.3f} {wz:.3f} 0 0 0</pose>')
        lines.append('      <link name="link">')
        lines.append('        <collision name="collision">')
        lines.append('          <geometry>')
        lines.append(f'            <box><size>{sx:.3f} {sy:.3f} {hz:.3f}</size></box>')
        lines.append('          </geometry>')
        lines.append('        </collision>')
        lines.append('        <visual name="visual">')
        lines.append('          <geometry>')
        lines.append(f'            <box><size>{sx:.3f} {sy:.3f} {hz:.3f}</size></box>')
        lines.append('          </geometry>')
        lines.append('        </visual>')
        lines.append('      </link>')
        lines.append('    </model>')

    lines.append('  </world>')
    lines.append('</sdf>')

    logger.info(f"Generated SDF with {len(boxes)} buildings")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    map_png="./assets/finalmap.png",
    out_dir="gz_world_out",
    resolution_m_per_px=0.2,
    seed=42,
):
    logger.info(f"Starting world generation from: {map_png}")
    logger.info(f"Output directory: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(map_png, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {map_png}")

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    H, W = img.shape[:2]
    logger.info(f"Image dimensions: {W}x{H} pixels")

    label_map, min_dist = segment_by_palette(img)
    mask_building, mask_roads = build_masks(label_map, min_dist, building_tol=12.0, road_tol=18.0)

    cv2.imwrite(os.path.join(out_dir, "mask_building.png"), mask_building)
    cv2.imwrite(os.path.join(out_dir, "mask_roads.png"), mask_roads)

    boxes = extract_building_boxes(
        building_mask_255=mask_building,
        roads_mask_255=mask_roads,
        min_area=800,
        min_side=20,
        max_side=300,
        max_aspect_ratio=3.5,
        max_road_fraction=0.02,
    )
    logger.info(f"Selected {len(boxes)} building boxes after road-overlap filtering")

    heightmap = Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L")
    heightmap_path = os.path.join(out_dir, "heightmap.png")
    heightmap.save(heightmap_path)
    logger.info(f"Created flat heightmap: {heightmap_path}")

    texture_root_path = os.path.join(out_dir, "finalmap.png")
    Image.open(map_png).save(texture_root_path)
    logger.info(f"Copied texture map: {texture_root_path}")

    write_ogre_material(out_dir=out_dir, texture_filename="finalmap.png")

    textures_dir = os.path.join(out_dir, "materials", "textures")
    os.makedirs(textures_dir, exist_ok=True)
    texture_material_path = os.path.join(textures_dir, "finalmap.png")
    shutil.copyfile(texture_root_path, texture_material_path)
    logger.info(f"Copied texture into: {texture_material_path}")

    sdf = make_world_sdf(
        W_px=W,
        H_px=H,
        boxes=boxes,
        resolution_m_per_px=resolution_m_per_px,
        seed=seed,
    )

    sdf_path = os.path.join(out_dir, "utm_world.sdf")
    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf)

    logger.info("World generation completed successfully")
    logger.info(f"Output directory: {os.path.abspath(out_dir)}")
    logger.info(f"Generated {len(boxes)} buildings")
    logger.info(f"SDF file: {sdf_path}")

    print("\n[SUCCESS] World generated")
    print(f"  • Output dir: {os.path.abspath(out_dir)}")
    print(f"  • Buildings: {len(boxes)}")
    print(f"  • SDF world: {sdf_path}")
    print(f"  • Heightmap: {heightmap_path}")
    print(f"  • Texture(root): {texture_root_path}")
    print(f"  • Material script: {os.path.join(out_dir,'materials','scripts','utm.material')}")
    print(f"  • Texture(material): {texture_material_path}")

    print("\n[RUN] Use these commands (important! run from out_dir):")
    print(f"  cd {os.path.abspath(out_dir)}")
    print("  export GAZEBO_MODEL_PATH=$PWD:$GAZEBO_MODEL_PATH")
    print("  export GAZEBO_RESOURCE_PATH=$PWD:$GAZEBO_RESOURCE_PATH")
    print("  gazebo --verbose utm_world.sdf")


if __name__ == "__main__":
    main()
