import os
import traceback
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from matplotlib.path import Path as GeoPath
import copy

class MeshGenerator():
    def __init__(self, rgb, dsm, dtm, mask, tree_boxes, height_scale=0.1):
        self.rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self.dsm = dsm.astype(np.float32)
        self.dtm = dtm.astype(np.float32)
        self.mask = mask
        assert self.rgb.shape[:2] == self.dsm.shape == self.dtm.shape == self.mask.shape, "Image dimensions must match!"
        self.height_scale = height_scale
        self.tree_boxes = tree_boxes
        self.tree_model_path = self._setup_tree_assets()
        
        # Load textures (with error handling)
        self.wall_texture = self._load_texture("walltex.jpg")
        self.roof_texture = self._load_texture("rooftex.jpg")

    def _load_texture(self, texture_name):
        """Load texture from local path or download from Hugging Face Hub"""

        try:
            from huggingface_hub import hf_hub_download
            print(f"Downloading texture {texture_name} from Hugging Face Hub...")
            
            # Download from HF Hub
            downloaded_path = hf_hub_download(
                repo_id="krdgomer/elevate3d-weights",
                filename=f"{texture_name}",
                cache_dir="hf_cache",
                force_download=False
            )
            
            return o3d.io.read_image(downloaded_path)
        
        except Exception as e:
            print(f"Failed to download texture {texture_name} from HF Hub: {e}")
            # Fallback to generated texture
            return self._create_fallback_texture()

    def _setup_tree_assets(self):
        """Download and setup tree assets from Hugging Face"""
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id="krdgomer/elevate3d-weights",
                filename="pine_tree.glb",
                cache_dir="hf_cache"
            )
        except Exception as e:
            print(f"HF Hub download failed: {str(e)}")
            return None

    def generate_tree_meshes(self, tree_boxes_df, tree_model_path, fixed_height=0.05):
        if tree_boxes_df is None or len(tree_boxes_df) == 0:
            return []

        try:
            if not tree_model_path or not os.path.exists(tree_model_path):
                raise FileNotFoundError(f"Tree model not found at {tree_model_path}")
            
            tree_model = o3d.io.read_triangle_mesh(tree_model_path,enable_post_processing=True)
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

            h, w = self.dtm.shape
            tree_meshes = []
            
            for _, row in tree_boxes_df.iterrows():
                center_x = int((row["xmin"] + row["xmax"]) / 2)
                center_y = int((row["ymin"] + row["ymax"]) / 2)

                if center_x >= w or center_y >= h:
                    continue

                base_z = self.dtm[center_y, center_x] * self.height_scale
                nx = center_x / w
                ny = center_y / h
                tree = copy.deepcopy(tree_model).translate((nx, ny, base_z))
                tree_meshes.append(tree)

            return tree_meshes

        except Exception as e:
            print(f"Error generating trees: {e}")
            traceback.print_exc()
            return []

    def generate_terrain_mesh(self):
        h, w = self.dtm.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        vertices = np.stack((x.flatten(), y.flatten(), self.dtm.flatten()), axis=1)
        vertices[:, 0] /= w
        vertices[:, 1] /= h
        vertices[:, 2] *= self.height_scale

        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                faces.append([idx, idx + 1, idx + w])
                faces.append([idx + 1, idx + w + 1, idx + w])

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces)
        )

        colors = self.rgb.reshape(-1, 3) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()
        return mesh

    def generate_building_meshes(self):
        building_meshes = []
        unique_ids = np.unique(self.mask)
        unique_ids = unique_ids[unique_ids > 0]
        h, w = self.dtm.shape

        # Get all building heights first to normalize
        building_heights = []
        for bid in unique_ids:
            building_mask = (self.mask == bid)
            dsm_values = self.dsm[building_mask]
            if len(dsm_values) > 0:
                avg_height = np.mean(dsm_values)
                building_heights.append(avg_height)
        
        if not building_heights:
            return building_meshes
        
        # Normalize heights to range [min_height, max_height]
        min_height = 0.01  # Minimum building height (normalized)
        max_height = 0.05  # Maximum building height (normalized)
        
        # Scale DSM values (0-255) to target height range
        min_dsm = np.min(building_heights)
        max_dsm = np.max(building_heights)
        
        for bid in unique_ids:
            region = (self.mask == bid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue

                epsilon = 1.0
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue

                mask_poly = np.zeros_like(region, dtype=np.uint8)
                cv2.drawContours(mask_poly, [approx], -1, 255, thickness=-1)
                building_area = (mask_poly == 255)
                
                if np.sum(building_area) < 10:
                    continue

                # Base height from DTM (ground level)
                base_height = np.mean(self.dtm[building_area]) * self.height_scale
                
                # Calculate building height from DSM (normalized to target range)
                dsm_values = self.dsm[building_area]
                avg_dsm = np.mean(dsm_values) if len(dsm_values) > 0 else min_dsm
                
                # Normalize height proportionally
                if max_dsm != min_dsm:  # Avoid division by zero
                    normalized_height = (avg_dsm - min_dsm) / (max_dsm - min_dsm)
                    height = min_height + (max_height - min_height) * normalized_height
                else:
                    height = (min_height + max_height) / 2  # Default if all buildings have same DSM
                
                footprint = approx[:, 0, :].astype(np.float32)
                footprint[:, 0] /= w
                footprint[:, 1] /= h

                # --- Rest of your existing mesh generation code ---
                bottom = np.column_stack((footprint, np.full(len(footprint), base_height)))
                top = np.column_stack((footprint, np.full(len(footprint), base_height + height)))
                vertices = np.vstack((bottom, top))

                faces = []
                uv_coords = []
                material_ids = []
                path = GeoPath(footprint)

                # ROOF (Delaunay triangulation)
                if len(footprint) >= 3:
                    tri = Delaunay(footprint)
                    for simplex in tri.simplices:
                        centroid = np.mean(footprint[simplex], axis=0)
                        if path.contains_point(centroid):
                            faces.append([
                                simplex[2] + len(footprint),
                                simplex[1] + len(footprint),
                                simplex[0] + len(footprint)
                            ])
                            # UV mapping for roof
                            for idx in simplex:
                                uv_x = (footprint[idx][0] - np.min(footprint[:,0])) / (np.max(footprint[:,0]) - np.min(footprint[:,0]))
                                uv_y = (footprint[idx][1] - np.min(footprint[:,1])) / (np.max(footprint[:,1]) - np.min(footprint[:,1]))
                                uv_coords.append([uv_x, uv_y])
                            material_ids.append(1)

                # WALLS
                for i in range(len(footprint)):
                    i_next = (i + 1) % len(footprint)
                    b1, b2 = i, i_next
                    t1, t2 = b1 + len(footprint), b2 + len(footprint)

                    faces.append([b1, b2, t2])
                    faces.append([b1, t2, t1])
                    
                    wall_height = height / max_height  # Normalized to tallest building
                    uv_coords.extend([
                        [0, 0], [1, 0], [1, wall_height],
                        [0, 0], [1, wall_height], [0, wall_height]
                    ])
                    material_ids.extend([0, 0])

                # Create mesh
                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices),
                    triangles=o3d.utility.Vector3iVector(faces)
                )
                mesh.textures = [self.wall_texture, self.roof_texture]
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coords)
                mesh.triangle_material_ids = o3d.utility.IntVector(material_ids)
                mesh.compute_vertex_normals()
                building_meshes.append(mesh)

        return building_meshes
    
    def visualize(self, save_path=None):
        terrain = self.generate_terrain_mesh()
        buildings = self.generate_building_meshes()
        trees = self.generate_tree_meshes(self.tree_boxes, self.tree_model_path) if self.tree_boxes is not None else []

        combined_mesh = [terrain] + buildings + trees

        if save_path:
            try:
                import trimesh
                # Convert all Open3D meshes to trimesh and combine
                tri_meshes = []
                for o3d_mesh in combined_mesh:
                    tri_mesh = trimesh.Trimesh(
                        vertices=np.asarray(o3d_mesh.vertices),
                        faces=np.asarray(o3d_mesh.triangles),
                        vertex_colors=np.asarray(o3d_mesh.vertex_colors)
                    )
                    tri_meshes.append(tri_mesh)
                
                # Combine all meshes
                scene = trimesh.Scene()
                for mesh in tri_meshes:
                    scene.add_geometry(mesh)
                
                # Export as GLB
                scene.export(save_path)
                return save_path
                
            except Exception as e:
                print(f"Error exporting with trimesh: {str(e)}")
                # Fall back to Open3D export
                return self.visualize(save_path)
        else:
            o3d.visualization.draw_geometries(combined_mesh, mesh_show_back_face=True)
            return None