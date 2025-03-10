import numpy as np
import cv2
import trimesh
import json
from shapely.geometry import Polygon

def load_image(path, grayscale=True, as_uint8=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Error: Could not read image at {path}")
    
    if as_uint8:
        return img.astype(np.uint8)  # OpenCV için uint8 formatı gerekiyor
    else:
        return img.astype(np.float32)  # DSM ve DTM için float32

def load_bounding_boxes(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_terrain(dtm, rgb_path):
    h, w = dtm.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    vertices = np.column_stack((x.ravel(), y.ravel(), dtm.ravel()))
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v1 = i * w + j
            v2 = v1 + 1
            v3 = v1 + w
            v4 = v3 + 1
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    terrain = trimesh.Trimesh(vertices=vertices, faces=faces)
    terrain.visual = trimesh.visual.TextureVisuals(image=rgb_path)
    return terrain

def extrude_buildings(building_mask, dsm, dtm):
    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    buildings = []
    for cnt in contours:
        cnt = cnt[:, 0, :]  # OpenCV konturlarını düzene sok

        # Konturu shapely Polygon nesnesine çevir
        polygon = Polygon(cnt)
        if not polygon.is_valid:
            continue  # Geçersiz poligonları atla

        # Bina yüksekliğini belirle
        min_x, min_y = cnt.min(axis=0)
        max_x, max_y = cnt.max(axis=0)
        building_height = dsm[min_y:max_y, min_x:max_x].max() - dtm[min_y:max_y, min_x:max_x].min()

        # 3D ekstrüzyon yap
        extruded = trimesh.creation.extrude_polygon(polygon, height=building_height)
        buildings.append(extruded)

    return buildings

def generate_3d_model(rgb_path, dsm_path, dtm_path, building_mask_path, output_path):
    rgb = load_image(rgb_path, grayscale=False)
    dsm = load_image(dsm_path).astype(np.float32)
    dtm = load_image(dtm_path).astype(np.float32)
    building_mask = load_image(building_mask_path, grayscale=True, as_uint8=True)
    
    terrain = generate_terrain(dtm, rgb_path)
    buildings = extrude_buildings(building_mask, dsm, dtm)
    
    scene = trimesh.Scene()
    scene.add_geometry(terrain)
    for building in buildings:
        scene.add_geometry(building)
    
    material_filename = output_path.replace(".obj", ".mtl")
    with open(material_filename, "w") as mtl_file:
        mtl_file.write(f"newmtl texture_material\n")
        mtl_file.write(f"map_Kd {rgb_path}\n")
    
    with open(output_path, "w") as obj_file:
        obj_file.write(f"mtllib {material_filename}\n")
    
    scene.export(output_path)

generate_3d_model("src/3dmodeling/rgb.png", "src/3dmodeling/dsm.png", "src/3dmodeling/dtm.png", "src/3dmodeling/building_mask.png", "output_model.obj")
