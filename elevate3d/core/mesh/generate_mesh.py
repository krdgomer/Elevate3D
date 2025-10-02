import traceback
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from elevate3d.core.mesh.texture_manager import TextureManager
from elevate3d.core.mesh.generate_building_mesh import BuildingMeshGenerator
from elevate3d.core.mesh.generate_tree_mesh import TreeMeshGenerator
from elevate3d.core.mesh.generate_terrain_mesh import TerrainMeshGenerator
from elevate3d.core.roof.predict_roof import SimpleRoofPredictor
from elevate3d.core.mesh.building import BuildingManager

class MeshGenerator:
    def __init__(self, rgb, dsm, dtm, mask, tree_boxes, height_scale=0.1):
        self.rgb = rgb
        self.dsm = dsm
        self.dtm = dtm
        self.mask = mask
        self.tree_boxes = tree_boxes
        self.height_scale = height_scale

        # Initialize components
        self.texture_manager = TextureManager()
        self.wall_texture = self.texture_manager.load_texture("walltex.jpg")
        self.roof_texture = self.texture_manager.load_texture("rooftex.jpg")
        self.roof_predictor = SimpleRoofPredictor()
        self.terrain_mesh_generator = TerrainMeshGenerator(self.dtm, self.rgb, self.height_scale)
        self.building_mesh_generator = BuildingMeshGenerator(
            self.rgb, self.dsm, self.dtm, self.mask,
            self.wall_texture, self.roof_texture, self.terrain_mesh_generator, self.height_scale
        )
        self.tree_mesh_generator = TreeMeshGenerator(self.dtm, self.height_scale)
        self.tree_model_path = self.tree_mesh_generator.setup_tree_assets()

        self.building_manager = BuildingManager(self.rgb, self.dsm, self.dtm, self.mask, self.roof_predictor, self.height_scale)
        

    def generate_building_meshes(self):
        self.building_manager.extract_buildings()
        return self.building_mesh_generator.generate_building_meshes(self.building_manager.buildings)

    def generate_tree_meshes(self, tree_boxes_df, tree_model_path, fixed_height=0.05):
        return self.tree_mesh_generator.generate_tree_meshes(tree_boxes_df,tree_model_path, fixed_height)

    def generate_terrain_mesh(self):
        return self.terrain_mesh_generator.generate_terrain_mesh()


    def visualize(self, save_path=None):
        terrain = self.generate_terrain_mesh()
        buildings = self.generate_building_meshes()
        trees = self.generate_tree_meshes(self.tree_boxes, self.tree_model_path) if self.tree_boxes is not None else []

        if save_path:
            try:
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
                [terrain] + buildings,
                mesh_show_back_face=True,
                mesh_show_wireframe=False
            )
            return None