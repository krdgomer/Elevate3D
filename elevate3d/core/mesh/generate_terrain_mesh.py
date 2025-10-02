import numpy as np
import open3d as o3d
from scipy import ndimage

class TerrainMeshGenerator:
    def __init__(self, dtm, rgb, height_scale=1.0, smooth_sigma=10, terrain_height_range=0.2):
        self.dtm = dtm.copy()
        self.rgb = rgb
        self.height_scale = height_scale
        self.smooth_sigma = smooth_sigma
        self.terrain_height_range = terrain_height_range  # Max terrain height variation
        self.h, self.w = dtm.shape
    
    def generate_terrain_mesh(self):
        h, w = self.h, self.w
        
        # Create grid coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_norm = x.astype(np.float32) / w
        y_norm = y.astype(np.float32) / h
        
        # Calculate height - PRESERVE the natural terrain variation
        min_dtm = np.min(self.dtm)
        max_dtm = np.max(self.dtm)
        
        if max_dtm != min_dtm:
            # Normalize to [0,1] but preserve the relative height differences
            normalized_height = (self.dtm - min_dtm) / (max_dtm - min_dtm)
            # Scale to desired height range (not too flat!)
            z = normalized_height * self.terrain_height_range
        else:
            z = np.zeros_like(self.dtm)
        
        z = z * self.height_scale
        
        print(f"Final terrain height range: {z.min():.4f} to {z.max():.4f}")


        
        # Create vertices
        vertices = np.stack((x_norm.flatten(), y_norm.flatten(), z.flatten()), axis=1).astype(np.float32)
        
        # Create faces
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                faces.append([idx, idx + w, idx + 1])
                faces.append([idx + 1, idx + w, idx + w + 1])
        
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        
        # Add colors
        if self.rgb.shape[:2] == (h, w):
            colors = self.rgb.reshape(-1, 3) / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        mesh.compute_vertex_normals()
        return mesh,z
    
