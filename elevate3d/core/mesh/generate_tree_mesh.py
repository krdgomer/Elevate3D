import os
import copy
import numpy as np
import open3d as o3d
import traceback
from elevate3d.utils.download_manager import DownloadManager
from elevate3d.core.mesh.building import Building


class TreeMeshGenerator:
    def __init__(self,height_scale=0.1):
        self.height_scale = height_scale
        self.download_manager = DownloadManager()

    def setup_tree_assets(self):
        """Download and setup tree assets from Hugging Face"""
        try:
            return self.download_manager.download_file("pine_tree.glb")
        except Exception as e:
            print(f"Failed to download tree assets: {e}")
            return None

    def generate_tree_meshes(self, z, tree_boxes_df, tree_model_path, buildings: list[Building], fixed_height=0.05):
        if tree_boxes_df is None or len(tree_boxes_df) == 0:
            return []

        try:
            if not tree_model_path or not os.path.exists(tree_model_path):
                raise FileNotFoundError(f"Tree model not found at {tree_model_path}")
            
            tree_model = o3d.io.read_triangle_mesh(tree_model_path, enable_post_processing=True)
            if not tree_model.has_vertices():
                raise ValueError("Loaded tree model has no vertices")
                
            # Prepare tree model
            tree_model.compute_vertex_normals()
            R = tree_model.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
            tree_model.rotate(R, center=tree_model.get_center())
            
            # Position and scale
            bbox = tree_model.get_axis_aligned_bounding_box()
            tree_offset = -bbox.get_min_bound()[2]
            tree_model.translate((0, 0, tree_offset))
            
            center_xy_offset = tree_model.get_axis_aligned_bounding_box().get_center()
            tree_model.translate((-center_xy_offset[0], -center_xy_offset[1], 0))
            
            scale_factor = fixed_height / bbox.get_extent()[2]
            tree_model.scale(scale_factor, center=(0, 0, 0))

            h, w = z.shape
            tree_meshes = []
            
            for _, row in tree_boxes_df.iterrows():
                center_x = int((row["xmin"] + row["xmax"]) / 2)
                center_y = int((row["ymin"] + row["ymax"]) / 2)

                if center_x >= w or center_y >= h:
                    continue

                # Check if the tree is inside any building
                is_inside_building = False
                for building in buildings:
                    if building.region is not None and building.region[center_y, center_x]:
                        is_inside_building = True
                        break

                if is_inside_building:
                    print(f"Tree at ({center_x}, {center_y}) is inside a building. Discarding.")
                    continue

                # Calculate tree position
                base_z = z[center_y, center_x] 
                nx = center_x / w
                ny = center_y / h
                tree = copy.deepcopy(tree_model).translate((nx, ny, base_z))
                tree_meshes.append(tree)

            return tree_meshes

        except Exception as e:
            print(f"Error generating trees: {e}")
            traceback.print_exc()
            return []