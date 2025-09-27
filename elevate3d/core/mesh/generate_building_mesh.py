import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path as GeoPath
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

class BuildingMeshGenerator:
    def __init__(self, rgb, dsm, dtm, mask, roof_predictor, wall_texture, roof_texture, terrain_generator, height_scale=0.2):
        self.rgb = rgb
        self.dsm = dsm
        self.dtm = dtm
        self.mask = mask
        self.roof_predictor = roof_predictor
        self.wall_texture = wall_texture
        self.roof_texture = roof_texture
        self.height_scale = height_scale
        self.terrain_generator = terrain_generator


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
        """Create building with gabled roof."""
        print("Creating gabled roof building")

        n_verts = len(footprint)

        # Find the longest edge to determine ridge direction
        edge_lengths = []
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            edge_length = np.linalg.norm(footprint[next_i] - footprint[i])
            edge_lengths.append(edge_length)

        longest_edge_idx = np.argmax(edge_lengths)

        # Get the endpoints of the longest edge
        ridge_start_idx = longest_edge_idx
        ridge_end_idx = (longest_edge_idx + 1) % n_verts
        longest_edge_start = footprint[ridge_start_idx]
        longest_edge_end = footprint[ridge_end_idx]

        # Calculate the centroid of the building footprint
        centroid = np.mean(footprint, axis=0)

        # Calculate the direction vector of the longest edge
        edge_direction = longest_edge_end - longest_edge_start
        edge_direction /= np.linalg.norm(edge_direction)  # Normalize the direction vector

        # Calculate the ridge line length (equal to the longest edge length)
        ridge_length = np.linalg.norm(longest_edge_end - longest_edge_start)

        # Calculate the ridge line passing through the centroid
        ridge_start = centroid - edge_direction * (ridge_length / 2)  # Half the ridge length in one direction
        ridge_end = centroid + edge_direction * (ridge_length / 2)  # Half the ridge length in the other direction

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

        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            wall_top_idx = i + n_verts
            next_wall_top_idx = next_i + n_verts

            # Create two triangles for each wall segment
            faces.append([wall_top_idx, next_wall_top_idx, ridge_start_idx])
            faces.append([wall_top_idx, ridge_start_idx, ridge_end_idx])

            # UV coordinates for roof
            uv_coords.extend([[0, 0], [1, 0], [0.5, 1]])
            uv_coords.extend([[0, 0], [0.5, 1], [1, 1]])
            material_ids.extend([1, 1])  # Roof material

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
        """Generate building meshes with roof classification."""
        building_meshes = []
        unique_ids = np.unique(self.mask)
        unique_ids = unique_ids[unique_ids > 0]
        h, w = self.dtm.shape

        # Get the EXACT same terrain parameters as terrain generator
        min_dtm = np.min(self.dtm)
        max_dtm = np.max(self.dtm)
        terrain_height_range = 0.1  # MUST match terrain generator's terrain_height_range
        height_scale = self.height_scale

        print(f"Building generator using: min_dtm={min_dtm:.4f}, max_dtm={max_dtm:.4f}, terrain_height_range={terrain_height_range}")

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

                # Prepare footprint
                footprint = approx[:, 0, :].astype(np.float32)
                footprint[:, 0] /= w
                footprint[:, 1] /= h

                # DEBUG: Check what terrain values we're working with in this area
                building_pixels = np.where(building_area)
                building_dtm_values = self.dtm[building_pixels]
                print(f"Building {bid}: DTM values in area: min={np.min(building_dtm_values):.2f}, max={np.max(building_dtm_values):.2f}, mean={np.mean(building_dtm_values):.2f}")

                # Get base height from terrain - use AVERAGE of building area, not just centroid
                building_pixels = np.where(building_area)
                building_dtm_values = self.dtm[building_pixels]
                
                # EXACT SAME CALCULATION AS TERRAIN GENERATOR
                if max_dtm != min_dtm:
                    # Normalize the AVERAGE DTM value of the building area
                    avg_dtm_value = np.mean(building_dtm_values)
                    normalized_height = (avg_dtm_value - min_dtm) / (max_dtm - min_dtm)
                    base_height = normalized_height * terrain_height_range
                else:
                    base_height = 0.0
                    
                base_height = base_height * height_scale

                # Calculate building height from DSM (relative to terrain)
                dsm_values = self.dsm[building_area]
                avg_dsm = np.mean(dsm_values) if len(dsm_values) > 0 else np.min(self.dsm)

                # Calculate height above terrain based on DSM-DTM difference
                avg_dtm = np.mean(building_dtm_values)
                height_above_terrain = max(0.01, (avg_dsm - avg_dtm) * 0.001)  # Scale appropriately

                # Alternatively, use normalized approach but ensure it's reasonable
                min_height_above_terrain = 0.01
                max_height_above_terrain = 0.05
                
                if max_dtm != min_dtm:
                    normalized_dsm = (avg_dsm - min_dtm) / (max_dtm - min_dtm)
                    height_above_terrain = min_height_above_terrain + (max_height_above_terrain - min_height_above_terrain) * normalized_dsm
                else:
                    height_above_terrain = (min_height_above_terrain + max_height_above_terrain) / 2

                height_above_terrain = max(0.01, min(height_above_terrain, 0.08))
                
                # Crop the RGB image
                y_coords, x_coords = np.where(building_area)
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                cropped_rgb = self.rgb[y_min:y_max + 1, x_min:x_max + 1]

                # Convert to image for roof predictor
                building_rgb_image = Image.fromarray(cropped_rgb)

                # Predict roof type
                predicted_class, confidence, all_probs = self.roof_predictor.predict(building_rgb_image)
                print(f"Building {bid}: Base={base_height:.4f}, Above={height_above_terrain:.4f}, Total={base_height + height_above_terrain:.4f}")

                # Generate mesh
                if predicted_class == "flat":
                    vertices, faces, uv_coords, material_ids = self.create_flat_roof_building(
                        footprint, base_height, height_above_terrain, h, w
                    )
                elif predicted_class == "gable":
                    vertices, faces, uv_coords, material_ids = self.create_gabled_roof_building(
                        footprint, base_height, height_above_terrain, h, w
                    )
                elif predicted_class == "hip":
                    vertices, faces, uv_coords, material_ids = self.create_hip_roof_building(
                        footprint, base_height, height_above_terrain, h, w
                    )
                else:
                    vertices, faces, uv_coords, material_ids = self.create_flat_roof_building(
                        footprint, base_height, height_above_terrain, h, w
                    )

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