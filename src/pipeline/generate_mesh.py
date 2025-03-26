import cv2
import numpy as np
import open3d as o3d

class MeshGenerator():
    def __init__(self, rgb_path, dsm_path, dtm_path, masks_path):
        self.rgb_path = rgb_path
        self.dsm_path = dsm_path
        self.dtm_path = dtm_path    
        self.masks_path = masks_path

    def generate_elevation_map(self):
        dsm = cv2.imread(self.dsm_path, cv2.IMREAD_GRAYSCALE)  # DSM
        dtm = cv2.imread(self.dtm_path, cv2.IMREAD_GRAYSCALE)  # DTM
        mask = cv2.imread(self.masks_path, cv2.IMREAD_GRAYSCALE)  # Labeled mask

        assert dsm.shape == dtm.shape == mask.shape, "Image dimensions must match!"

        final_elevation = dtm.copy()

        # Get unique building labels (excluding background 0)
        unique_buildings = np.unique(mask)
        unique_buildings = unique_buildings[unique_buildings > 0]  # Remove background (0)

        # Process each building separately
        for building_id in unique_buildings:
            # Get mask for the current building
            building_mask = (mask == building_id)

            # Compute mean DSM elevation for this building
            mean_elevation = np.mean(dsm[building_mask])

            # Assign the mean elevation to all pixels of this building in the DTM
            final_elevation[building_mask] = int(mean_elevation)

        return final_elevation


    def generate_terrain_mesh(self):
        print("Generating terrain mesh...")
        rgb = rgb = cv2.cvtColor(cv2.imread(self.rgb_path), cv2.COLOR_BGR2RGB)
        dtm = self.generate_elevation_map()

        # Create grid
        h, w = dtm.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        x_scaled = (x / w) - 0.5
        y_scaled = (y / h) - 0.5

        # Flatten the arrays
        x = x.flatten()
        y = y.flatten()
        z = dtm.flatten()

        # Create vertices
        vertices = np.vstack((x, y, z)).T

        # Create faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                faces.append([idx, idx + 1, idx + w])
                faces.append([idx + 1, idx + w + 1, idx + w])

        faces = np.array(faces)

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Optionally, add vertex colors
        colors = rgb.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Compute vertex normals for shading
        mesh.compute_vertex_normals()

        u = x / (w - 1)
        v = y / (h - 1)
        uvs = np.vstack((u, v)).T

        mesh.textures = [o3d.geometry.Image(rgb)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))

        # Visualize the mesh with shading
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    mesh_generator = MeshGenerator("src/3dmodeling/test_images/rgb.png", "src/3dmodeling/test_images/dsm.png", 
                                  "src/3dmodeling/test_images/output_dtm.png", "src/3dmodeling/test_images/labeled_mask.png")
    mesh = mesh_generator.generate_terrain_mesh()