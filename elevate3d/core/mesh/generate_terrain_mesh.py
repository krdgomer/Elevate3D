import numpy as np
import open3d as o3d

class TerrainMeshGenerator:
    def __init__(self, dtm, rgb, height_scale=0.1):
        self.dtm = dtm
        self.rgb = rgb
        self.height_scale = height_scale

    def generate_terrain_mesh(self):
        h, w = self.dtm.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))


        vertices = np.stack((x.flatten(), y.flatten(), self.dtm.flatten()), axis=1).astype(np.float32)
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