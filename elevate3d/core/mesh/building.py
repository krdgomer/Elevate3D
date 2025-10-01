import numpy as np
import cv2
from PIL import Image

class Building:
    def __init__(self,bid):
        self.id = bid
        self.footprint = None          # Polygon vertices
        self.region = None             # Array where building area is 1
        self.raw_height = 0.0          # From DSM
        self.base_height = 0.0         # From terrain
        self.roof_type = "unknown"     # flat/gable/hip
        self.roof_confidence = 0.0
        self.roof_color = None         # Dominant roof color
        self.position = None           # Centroid
        self.area = 0.0
        self.building_type = "unknown" # residential/commercial/etc.
        self.mesh = None
        self.bounding_box = None
        self.should_keep = True
        self.is_regular_shape = False

class BuildingManager:
    def __init__(self, rgb, dsm, dtm, mask, roof_predictor):
        self.buildings = []
        self.rgb = rgb
        self.dsm = dsm
        self.dtm = dtm
        self.mask = mask
        self.roof_predictor = roof_predictor

    def extract_buildings(self):
        unique_ids = np.unique(self.mask)
        unique_ids = unique_ids[unique_ids > 0] #Discard background
        
        for bid in unique_ids:
            
            building = Building(bid)
            
            # Extract footprint
            building.footprint,building.region = self._get_footprint(bid)
            building.position = self._get_centroid(building.footprint)
            building.area = self._calculate_area(building.footprint)
            
            # Height analysis
            building.raw_height = self._get_height_from_dsm(bid)
            building.base_height = self._get_terrain_height(bid)
            
            # Roof analysis  
            building.roof_type, building.roof_confidence = self._predict_roof_type(bid,building.region)
            building.roof_color = self._extract_roof_color(bid)
            
            self.buildings.append(building)

    def _get_footprint(self, bid):
        """Extract the footprint (polygon) of the building with ID `bid`."""
        # Create a binary mask for the building
        building_mask = (self.mask == bid).astype(np.uint8)
        
        # Find contours of the building
        contours, _ = cv2.findContours(building_mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Assume the largest contour is the building footprint
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour.squeeze(),building_mask  # Return as a 2D array of (x, y) points
        return None

    def _get_centroid(self, footprint):
        """Calculate the centroid of the building footprint."""
        if footprint is None or len(footprint) == 0:
            return None
        moments = cv2.moments(footprint)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return np.array([cx, cy])

    def _calculate_area(self, footprint):
        """Calculate the area of the building footprint."""
        if footprint is None or len(footprint) < 3:
            return 0.0
        return cv2.contourArea(footprint)

    def _get_height_from_dsm(self, bid):
        """Calculate the raw height of the building from the DSM."""
        building_mask = (self.mask == bid)
        building_heights = self.dsm[building_mask]
        if len(building_heights) == 0:
            return 0.0
        return np.mean(building_heights)

    def _get_terrain_height(self, bid):
        """Calculate the base height of the building from the DTM."""
        building_mask = (self.mask == bid)
        terrain_heights = self.dtm[building_mask]
        if len(terrain_heights) == 0:
            return 0.0
        return np.mean(terrain_heights)

    def _predict_roof_type(self, bid, building_region):
        """Predict the roof type and confidence for the building."""
        y_coords, x_coords = np.where(building_region)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        cropped_rgb = self.rgb[y_min:y_max + 1, x_min:x_max + 1]
        building_rgb_image = Image.fromarray(cropped_rgb)
        roof_type, confidence,all_probs = self.roof_predictor.predict(building_rgb_image)
        return roof_type, confidence

    def _extract_roof_color(self, bid):
        """Extract the dominant roof color of the building."""
        building_mask = (self.mask == bid)
        building_pixels = self.rgb[building_mask]
        if len(building_pixels) == 0:
            return None
        # Calculate the mean color of the roof
        mean_color = np.mean(building_pixels, axis=0)
        return mean_color.astype(np.uint8)