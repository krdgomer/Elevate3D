import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path as GeoPath
from PIL import Image
import open3d as o3d
from elevate3d.core.mesh.building import Building

class BuildingMeshGenerator:
    def __init__(self, rgb, dsm, dtm, mask, wall_texture, roof_texture):
        self.rgb = rgb
        self.dsm = dsm
        self.dtm = dtm
        self.mask = mask
        self.wall_texture = wall_texture
        self.roof_texture = roof_texture

    def generate_building(self, building: Building, image_width, image_height):
        """Generate a 3D mesh for a single building."""
        if building.area < 10:
            print(f"Building {building.id} too small, skipping.")
            return None

        roof_type_methods = {
            "flat": self.create_flat_roof_building,
            "gable": self.create_gabled_roof_building,
            "hip": self.create_hip_roof_building,
            "complex:": self.create_flat_roof_building,
            "pyramid": self.create_hip_roof_building
        }
        create_roof_method = roof_type_methods.get(building.roof_type, self.create_flat_roof_building)
        vertices, faces, uv_coordinates, material_ids = create_roof_method(
            building.normalized_footprint, building.base_height, building.raw_height, image_height, image_width
        )

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        mesh.textures = [self.wall_texture, self.roof_texture]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coordinates)
        mesh.triangle_material_ids = o3d.utility.IntVector(material_ids)
        mesh.compute_vertex_normals()

        return mesh

    def _create_wall_faces(self, n_verts, faces, uv_coords, material_ids):
        """Generate wall faces for a building footprint."""
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            b1, b2 = i, next_i
            t1, t2 = b1 + n_verts, b2 + n_verts

            faces.append([b1, b2, t2])
            faces.append([b1, t2, t1])
            
            wall_height_norm = 1.0
            uv_coords.extend([
                [0, 0], [1, 0], [1, wall_height_norm],
                [0, 0], [1, wall_height_norm], [0, wall_height_norm]
            ])
            material_ids.extend([0, 0])

    def create_flat_roof_building(self, footprint, base_height, building_height, h, w):
        """Create building with flat roof."""
        print("Creating flat roof building")
        
        n_verts = len(footprint)
        bottom = np.column_stack((footprint, np.full(n_verts, base_height)))
        top = np.column_stack((footprint, np.full(n_verts, base_height + building_height)))
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
                        simplex[2] + n_verts,
                        simplex[1] + n_verts,
                        simplex[0] + n_verts
                    ])
                    for idx in simplex:
                        uv_x = (footprint[idx][0] - np.min(footprint[:,0])) / (np.max(footprint[:,0]) - np.min(footprint[:,0])) if np.max(footprint[:,0]) != np.min(footprint[:,0]) else 0.5
                        uv_y = (footprint[idx][1] - np.min(footprint[:,1])) / (np.max(footprint[:,1]) - np.min(footprint[:,1])) if np.max(footprint[:,1]) != np.min(footprint[:,1]) else 0.5
                        uv_coords.append([uv_x, uv_y])
                    material_ids.append(1)

        # WALLS
        self._create_wall_faces(n_verts, faces, uv_coords, material_ids)

        return vertices, faces, uv_coords, material_ids

    def create_gabled_roof_building(self, footprint, base_height, building_height, h, w):
        """Create building with gabled roof."""
        print("Creating gabled roof building")

        n_verts = len(footprint)

        # Find longest edge for ridge direction
        edge_lengths = []
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            edge_length = np.linalg.norm(footprint[next_i] - footprint[i])
            edge_lengths.append(edge_length)

        longest_edge_idx = np.argmax(edge_lengths)
        ridge_start_idx = longest_edge_idx
        ridge_end_idx = (longest_edge_idx + 1) % n_verts
        
        longest_edge_start = footprint[ridge_start_idx]
        longest_edge_end = footprint[ridge_end_idx]

        # Calculate ridge line through centroid
        centroid = np.mean(footprint, axis=0)
        edge_direction = longest_edge_end - longest_edge_start
        edge_direction /= np.linalg.norm(edge_direction)
        ridge_length = np.linalg.norm(longest_edge_end - longest_edge_start)

        ridge_start = centroid - edge_direction * (ridge_length / 2)
        ridge_end = centroid + edge_direction * (ridge_length / 2)

        # Heights
        wall_height = building_height * 0.7
        ridge_height = building_height * 1.2

        # Build vertices
        bottom_vertices = np.column_stack((footprint, np.full(n_verts, base_height)))
        wall_top_vertices = np.column_stack((footprint, np.full(n_verts, base_height + wall_height)))
        ridge_vertices = np.array([
            [ridge_start[0], ridge_start[1], base_height + ridge_height],
            [ridge_end[0], ridge_end[1], base_height + ridge_height]
        ])

        vertices = np.vstack([bottom_vertices, wall_top_vertices, ridge_vertices])

        faces = []
        uv_coords = []
        material_ids = []

        # WALLS
        self._create_wall_faces(n_verts, faces, uv_coords, material_ids)

        # ROOF FACES
        ridge_start_idx = 2 * n_verts
        ridge_end_idx = 2 * n_verts + 1

        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            wall_top_idx = i + n_verts
            next_wall_top_idx = next_i + n_verts

            faces.append([wall_top_idx, next_wall_top_idx, ridge_start_idx])
            faces.append([wall_top_idx, ridge_start_idx, ridge_end_idx])

            uv_coords.extend([[0, 0], [1, 0], [0.5, 1]])
            uv_coords.extend([[0, 0], [0.5, 1], [1, 1]])
            material_ids.extend([1, 1])

        return vertices, faces, uv_coords, material_ids

    def create_hip_roof_building(self, footprint, base_height, building_height, h, w):
        """Create building with hip roof (pyramid-like)."""
        print("Creating hip roof building")

        n_verts = len(footprint)
        centroid = np.mean(footprint, axis=0)
        
        # Heights
        wall_height = building_height * 0.6
        peak_height = building_height * 1.1
        
        # Build vertices
        bottom_vertices = np.column_stack((footprint, np.full(n_verts, base_height)))
        wall_top_vertices = np.column_stack((footprint, np.full(n_verts, base_height + wall_height)))
        peak_vertex = np.array([[centroid[0], centroid[1], base_height + peak_height]])
        
        vertices = np.vstack([bottom_vertices, wall_top_vertices, peak_vertex])
        
        faces = []
        uv_coords = []
        material_ids = []
        peak_idx = 2 * n_verts
        
        # WALLS
        self._create_wall_faces(n_verts, faces, uv_coords, material_ids)
        
        # ROOF FACES
        for i in range(n_verts):
            next_i = (i + 1) % n_verts
            wall_top_i = i + n_verts
            wall_top_next = next_i + n_verts
            
            faces.append([wall_top_i, wall_top_next, peak_idx])
            uv_coords.extend([[0, 0], [1, 0], [0.5, 1]])
            material_ids.append(1)
        
        return vertices, faces, uv_coords, material_ids

    def generate_building_meshes(self, buildings):
        """Generate building meshes with roof classification."""
        building_meshes = []
        h, w = self.dsm.shape

        print("Generating building meshes on a flat plane.")

        for building in buildings:
            mesh = self.generate_building(building, w, h)
            if mesh is not None:
                building_meshes.append(mesh)

        return building_meshes