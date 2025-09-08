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
    
    def _create_fallback_texture(self):
        """Create a simple fallback texture"""
        # Create a simple colored texture
        texture_array = np.ones((64, 64, 3), dtype=np.uint8) * 128
        return o3d.geometry.Image(texture_array)

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

    def classify_roof_type(self, building_mask, dsm_values):
        """
        Classify roof type based on height variation within building footprint
        Returns: 'flat', 'gabled', 'hip', or 'complex'
        """
        if len(dsm_values) < 10:  # Too few points
            return 'flat'
        
        # Statistical analysis of height variation
        height_std = np.std(dsm_values)
        height_range = np.max(dsm_values) - np.min(dsm_values)
        height_mean = np.mean(dsm_values)
        
        # Get building dimensions for analysis
        y_coords, x_coords = np.where(building_mask)
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
        
        # Classification logic based on height variation and geometry
        if height_std < 0.5 and height_range < 2.0:
            return 'flat'
        elif height_std < 2.0 and aspect_ratio > 1.5:
            # Check for ridgeline pattern (characteristic of gabled roofs)
            if self._detect_ridgeline_pattern(building_mask, dsm_values):
                return 'gabled'
            else:
                return 'hip'
        elif height_std < 3.0:
            return 'hip'
        else:
            return 'complex'
    
    def _detect_ridgeline_pattern(self, building_mask, dsm_values):
        """
        Simple heuristic to detect if building has a ridgeline (gabled roof pattern)
        """
        # Get coordinates of building pixels
        y_coords, x_coords = np.where(building_mask)
        
        if len(y_coords) < 20:  # Too small for analysis
            return False
        
        # Find the direction of maximum variance (potential ridge direction)
        coords = np.column_stack([x_coords, y_coords])
        cov_matrix = np.cov(coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Principal direction (ridge direction)
        principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Project points onto the perpendicular direction (across ridge)
        perpendicular_direction = np.array([-principal_direction[1], principal_direction[0]])
        
        # Check if heights follow a triangular pattern across the ridge
        projections = np.dot(coords, perpendicular_direction)
        
        # Sort by projection and check height pattern
        sorted_indices = np.argsort(projections)
        sorted_heights = dsm_values[sorted_indices]
        
        # Simple check: heights should be lower at edges, higher in middle
        n = len(sorted_heights)
        edge_height = (np.mean(sorted_heights[:n//4]) + np.mean(sorted_heights[-n//4:])) / 2
        middle_height = np.mean(sorted_heights[n//4:3*n//4])
        
        return middle_height > edge_height + 1.0  # Ridge is at least 1 unit higher than edges

    def create_flat_roof_building(self, footprint, base_height, building_height, h, w):
        """Create building with flat roof (your existing logic)"""
        print("Creating flat roof building")
        bottom = np.column_stack((footprint, np.full(len(footprint), base_height)))
        top = np.column_stack((footprint, np.full(len(footprint), base_height + building_height)))
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
                        uv_x = (footprint[idx][0] - np.min(footprint[:,0])) / (np.max(footprint[:,0]) - np.min(footprint[:,0])) if np.max(footprint[:,0]) != np.min(footprint[:,0]) else 0.5
                        uv_y = (footprint[idx][1] - np.min(footprint[:,1])) / (np.max(footprint[:,1]) - np.min(footprint[:,1])) if np.max(footprint[:,1]) != np.min(footprint[:,1]) else 0.5
                        uv_coords.append([uv_x, uv_y])
                    material_ids.append(1)  # Roof material

        # WALLS
        max_height = 0.05  # Your max height for normalization
        for i in range(len(footprint)):
            i_next = (i + 1) % len(footprint)
            b1, b2 = i, i_next
            t1, t2 = b1 + len(footprint), b2 + len(footprint)

            faces.append([b1, b2, t2])
            faces.append([b1, t2, t1])
            
            wall_height_norm = building_height / max_height
            uv_coords.extend([
                [0, 0], [1, 0], [1, wall_height_norm],
                [0, 0], [1, wall_height_norm], [0, wall_height_norm]
            ])
            material_ids.extend([0, 0])  # Wall material

        return vertices, faces, uv_coords, material_ids

    def create_gabled_roof_building(self, footprint, base_height, building_height, h, w):
        """Create building with gabled roof"""
        print("Creating gabled roof building")

        n_verts = len(footprint)
        
        # Find the longest edge to determine ridge direction
        edge_lengths = []
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            edge_length = np.linalg.norm(footprint[next_i] - footprint[i])
            edge_lengths.append(edge_length)
        
        longest_edge_idx = np.argmax(edge_lengths)
        
        # Create ridge line parallel to longest edge
        ridge_start_idx = longest_edge_idx
        ridge_end_idx = (longest_edge_idx + 1) % n_verts
        
        # Calculate ridge line endpoints (offset inward)
        ridge_offset = 0.1  # 10% inward from edges
        ridge_start = footprint[ridge_start_idx] + ridge_offset * (footprint[ridge_end_idx] - footprint[ridge_start_idx])
        ridge_end = footprint[ridge_end_idx] - ridge_offset * (footprint[ridge_end_idx] - footprint[ridge_start_idx])
        
        # Calculate ridge height (higher than walls)
        wall_height = building_height * 0.7
        ridge_height = building_height * 1.2
        
        # Bottom vertices (foundation)
        bottom_vertices = np.column_stack((footprint, np.full(n_verts, base_height)))
        
        # Wall top vertices
        wall_top_vertices = np.column_stack((footprint, np.full(n_verts, base_height + wall_height)))
        
        # Ridge vertices
        ridge_vertices = np.array([
            [ridge_start[0], ridge_start[1], base_height + ridge_height],
            [ridge_end[0], ridge_end[1], base_height + ridge_height]
        ])
        
        vertices = np.vstack([bottom_vertices, wall_top_vertices, ridge_vertices])
        
        faces = []
        uv_coords = []
        material_ids = []
        
        # WALLS
        max_height = 0.05
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            b1, b2 = i, next_i
            t1, t2 = b1 + n_verts, b2 + n_verts

            faces.append([b1, b2, t2])
            faces.append([b1, t2, t1])
            
            wall_height_norm = wall_height / max_height
            uv_coords.extend([
                [0, 0], [1, 0], [1, wall_height_norm],
                [0, 0], [1, wall_height_norm], [0, wall_height_norm]
            ])
            material_ids.extend([0, 0])  # Wall material
        
        # ROOF FACES (connecting walls to ridge)
        ridge_start_idx = 2 * n_verts
        ridge_end_idx = 2 * n_verts + 1
        
        # Create roof triangles connecting wall edges to ridge
        for i in range(n_verts):
            wall_top_idx = i + n_verts
            next_wall_top_idx = ((i + 1) % n_verts) + n_verts
            
            # Determine which ridge point is closer
            wall_pos = vertices[wall_top_idx][:2]
            dist_to_start = np.linalg.norm(wall_pos - ridge_vertices[0][:2])
            dist_to_end = np.linalg.norm(wall_pos - ridge_vertices[1][:2])
            
            closer_ridge_idx = ridge_start_idx if dist_to_start < dist_to_end else ridge_end_idx
            
            # Create roof triangle
            faces.append([wall_top_idx, next_wall_top_idx, closer_ridge_idx])
            
            # UV coordinates for roof
            uv_coords.extend([[0, 0], [1, 0], [0.5, 1]])
            material_ids.append(1)  # Roof material
        
        return vertices, faces, uv_coords, material_ids

    def create_hip_roof_building(self, footprint, base_height, building_height, h, w):
        """Create building with hip roof (pyramid-like)"""
        print("Creating hip roof building")

        n_verts = len(footprint)
        
        # Calculate building centroid for hip peak
        centroid = np.mean(footprint, axis=0)
        
        wall_height = building_height * 0.6
        peak_height = building_height * 1.1
        
        # Bottom vertices
        bottom_vertices = np.column_stack((footprint, np.full(n_verts, base_height)))
        
        # Wall top vertices  
        wall_top_vertices = np.column_stack((footprint, np.full(n_verts, base_height + wall_height)))
        
        # Peak vertex
        peak_vertex = np.array([[centroid[0], centroid[1], base_height + peak_height]])
        
        vertices = np.vstack([bottom_vertices, wall_top_vertices, peak_vertex])
        
        faces = []
        uv_coords = []
        material_ids = []
        peak_idx = 2 * n_verts
        
        # WALLS
        max_height = 0.05
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            b1, b2 = i, next_i
            t1, t2 = b1 + n_verts, b2 + n_verts

            faces.append([b1, b2, t2])
            faces.append([b1, t2, t1])
            
            wall_height_norm = wall_height / max_height
            uv_coords.extend([
                [0, 0], [1, 0], [1, wall_height_norm],
                [0, 0], [1, wall_height_norm], [0, wall_height_norm]
            ])
            material_ids.extend([0, 0])
        
        # ROOF FACES (connecting wall edges to peak)
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            wall_top_i = i + n_verts
            wall_top_next = next_i + n_verts
            
            faces.append([wall_top_i, wall_top_next, peak_idx])
            uv_coords.extend([[0, 0], [1, 0], [0.5, 1]])
            material_ids.append(1)  # Roof material
        
        return vertices, faces, uv_coords, material_ids

    def generate_building_meshes(self):
        """Enhanced building mesh generation with roof classification"""
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
        
        # Normalize heights
        min_height = 0.01
        max_height = 0.05
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

                # Base height from DTM
                base_height = np.mean(self.dtm[building_area]) * self.height_scale
                
                # Calculate normalized building height
                dsm_values = self.dsm[building_area]
                avg_dsm = np.mean(dsm_values) if len(dsm_values) > 0 else min_dsm
                
                if max_dsm != min_dsm:
                    normalized_height = (avg_dsm - min_dsm) / (max_dsm - min_dsm)
                    height = min_height + (max_height - min_height) * normalized_height
                else:
                    height = (min_height + max_height) / 2
                
                # ROOF CLASSIFICATION - This is the key addition!
                roof_type = self.classify_roof_type(building_area, dsm_values)
                print(f"Building {bid}: roof type = {roof_type}")
                
                footprint = approx[:, 0, :].astype(np.float32)
                footprint[:, 0] /= w
                footprint[:, 1] /= h

                # Generate mesh based on classified roof type
                if roof_type == 'flat':
                    vertices, faces, uv_coords, material_ids = self.create_flat_roof_building(
                        footprint, base_height, height, h, w
                    )
                elif roof_type == 'gabled':
                    vertices, faces, uv_coords, material_ids = self.create_gabled_roof_building(
                        footprint, base_height, height, h, w
                    )
                elif roof_type == 'hip':
                    vertices, faces, uv_coords, material_ids = self.create_hip_roof_building(
                        footprint, base_height, height, h, w
                    )
                else:  # complex - fall back to flat for now
                    vertices, faces, uv_coords, material_ids = self.create_flat_roof_building(
                        footprint, base_height, height, h, w
                    )

                # Create mesh with classified roof geometry
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

    # Keep all your other methods (generate_tree_meshes, generate_terrain_mesh, visualize) unchanged
    def generate_tree_meshes(self, tree_boxes_df, tree_model_path, fixed_height=0.05):
        # Your existing tree generation code - unchanged
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
        # Your existing terrain generation - unchanged
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

    def visualize(self, save_path=None):
        # Your existing visualization code - unchanged
        terrain = self.generate_terrain_mesh()
        buildings = self.generate_building_meshes()
        trees = self.generate_tree_meshes(self.tree_boxes, self.tree_model_path) if self.tree_boxes is not None else []

        if save_path:
            try:
                import trimesh
                from PIL import Image
                import numpy as np
                
                scene = trimesh.Scene()
                
                def convert_mesh(o3d_mesh):
                    mesh = trimesh.Trimesh(
                        vertices=np.asarray(o3d_mesh.vertices),
                        faces=np.asarray(o3d_mesh.triangles),
                    )
                    
                    if o3d_mesh.has_vertex_colors():
                        mesh.visual.vertex_colors = np.asarray(o3d_mesh.vertex_colors)
                    
                    if o3d_mesh.has_triangle_uvs() and o3d_mesh.textures:
                        texture_array = np.asarray(o3d_mesh.textures[0])
                        texture_image = Image.fromarray(texture_array)
                        uv = np.asarray(o3d_mesh.triangle_uvs)
                        mesh.visual = trimesh.visual.TextureVisuals(
                            uv=uv,
                            image=texture_image
                        )
                    return mesh
                
                scene.add_geometry(convert_mesh(terrain))
                
                for building in buildings:
                    if building.has_triangle_uvs() and len(building.textures) >= 2:
                        triangles = np.asarray(building.triangles)
                        material_ids = np.asarray(building.triangle_material_ids)
                        uvs = np.asarray(building.triangle_uvs)
                        
                        valid_indices = min(len(triangles), len(material_ids), len(uvs))
                        triangles = triangles[:valid_indices]
                        material_ids = material_ids[:valid_indices]
                        uvs = uvs[:valid_indices]
                        
                        wall_mask = (material_ids == 0)
                        if np.any(wall_mask):
                            walls = o3d.geometry.TriangleMesh()
                            walls.vertices = building.vertices
                            walls.triangles = o3d.utility.Vector3iVector(triangles[wall_mask])
                            walls.triangle_uvs = o3d.utility.Vector2dVector(uvs[wall_mask])
                            walls.textures = [self.wall_texture]
                            scene.add_geometry(convert_mesh(walls))
                        
                        roof_mask = (material_ids == 1)
                        if np.any(roof_mask):
                            roof = o3d.geometry.TriangleMesh()
                            roof.vertices = building.vertices
                            roof.triangles = o3d.utility.Vector3iVector(triangles[roof_mask])
                            roof.triangle_uvs = o3d.utility.Vector2dVector(uvs[roof_mask])
                            roof.textures = [self.roof_texture]
                            scene.add_geometry(convert_mesh(roof))
                    else:
                        scene.add_geometry(convert_mesh(building))
                
                for tree in trees:
                    scene.add_geometry(convert_mesh(tree))
                
                scene.export(save_path)
                return save_path
                
            except Exception as e:
                print(f"Error saving model: {e}")
                traceback.print_exc()
                return None
        else:
            o3d.visualization.draw_geometries(
                [terrain] + buildings + trees,
                mesh_show_back_face=True,
                mesh_show_wireframe=False
            )
            return None