import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from matplotlib.path import Path


class MeshGenerator():
    def __init__(self, rgb_path, dsm_path, dtm_path, masks_path, height_scale=0.1):
        self.rgb_path = rgb_path
        self.dsm_path = dsm_path
        self.dtm_path = dtm_path    
        self.masks_path = masks_path
        self.height_scale = height_scale  # Scale factor to reduce building heights

    def load_data(self):
        self.rgb = cv2.cvtColor(cv2.imread(self.rgb_path), cv2.COLOR_BGR2RGB)
        self.dsm = cv2.imread(self.dsm_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        self.dtm = cv2.imread(self.dtm_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        self.mask = cv2.imread(self.masks_path, cv2.IMREAD_UNCHANGED)

        assert self.rgb.shape[:2] == self.dsm.shape == self.dtm.shape == self.mask.shape, "Image dimensions must match!"

    def generate_terrain_mesh(self):
        h, w = self.dtm.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Flatten arrays
        vertices = np.stack((x.flatten(), y.flatten(), self.dtm.flatten()), axis=1)
        vertices[:, 0] /= w
        vertices[:, 1] /= h
        vertices[:, 2] *= self.height_scale  # Scale terrain height to match buildings

        # Generate faces
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

        # Add vertex color from RGB
        colors = self.rgb.reshape(-1, 3) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()

        return mesh

    def generate_building_meshes(self):
        building_meshes = []
        unique_ids = np.unique(self.mask)
        unique_ids = unique_ids[unique_ids > 0]

        h, w = self.dtm.shape

        for bid in unique_ids:
            region = (self.mask == bid).astype(np.uint8) * 255

            # Find all contours (external + internal holes if any)
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue

                # Smooth contour while keeping shape — this reduces jagged edges
                epsilon = 1.0  # adjust if needed
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue

                # Get height values
                mask_poly = np.zeros_like(region, dtype=np.uint8)
                cv2.drawContours(mask_poly, [approx], -1, 255, thickness=-1)

                building_area = (mask_poly == 255)
                if np.sum(building_area) < 10:
                    continue

                base_height = np.mean(self.dtm[building_area]) * self.height_scale
                height_diff = self.dsm[building_area] - self.dtm[building_area]
                height = np.median(height_diff) * self.height_scale
                height = max(0.005, min(height, 0.05))

                # Prepare footprint (normalize coordinates)
                footprint = approx[:, 0, :].astype(np.float32)
                footprint[:, 0] /= w
                footprint[:, 1] /= h

                bottom = np.column_stack((footprint, np.full(len(footprint), base_height)))
                top = np.column_stack((footprint, np.full(len(footprint), base_height + height)))
                vertices = np.vstack((bottom, top))

                faces = []
                path = Path(footprint)

                tri = Delaunay(footprint)
                for tri_indices in tri.simplices:
                    pts = footprint[tri_indices]
                    centroid = np.mean(pts, axis=0)

                    if path.contains_point(centroid):  # Keep only triangles inside the footprint
                        # Bottom face (CCW)
                        faces.append([tri_indices[0], tri_indices[1], tri_indices[2]])
                        # Top face (CW)
                        offset = len(footprint)
                        faces.append([
                            tri_indices[2] + offset,
                            tri_indices[1] + offset,
                            tri_indices[0] + offset
                        ])

                # Side walls
                for i in range(len(footprint)):
                    i_next = (i + 1) % len(footprint)
                    b1, b2 = i, i_next
                    t1, t2 = b1 + len(footprint), b2 + len(footprint)
                    faces += [
                        [b1, b2, t2],
                        [b1, t2, t1]
                    ]

                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices),
                    triangles=o3d.utility.Vector3iVector(faces)
                )

                color_intensity = min(1.0, height * 10)
                mesh.paint_uniform_color([0.8, 0.8, color_intensity])
                mesh.compute_vertex_normals()
                building_meshes.append(mesh)

        return building_meshes


    def visualize(self):
        self.load_data()
        terrain = self.generate_terrain_mesh()
        buildings = self.generate_building_meshes()
        o3d.visualization.draw_geometries([terrain] + buildings, mesh_show_back_face=True)