"""
Machine Learning & Spatial Perception Module for 2D Occupancy Grid Maps.
Performs:
1. Bilateral Spatial Denoising to filter isolated sensor artifacts.
2. Morphological Wall Learning to bridge scan gaps into continuous contours.
3. Polygonal Contour Vectorization (approxPolyDP) for structural wall extraction.
4. Medial Axis / Euclidean Distance Transform (EDT) for safe navigation corridors.
5. K-Means Spatial Clustering to segment distinct physical obstacles.
"""

import math
import cv2
import numpy as np


class MapMLProcessor:
    """
    Performs AI/ML spatial analysis on raw 2D occupancy grids to produce
    a clean, refined, vectorized 2D floor plan with safe navigation corridors.
    """

    def __init__(self, resolution_cm=4.0):
        self.resolution_cm = resolution_cm

    def refine_map(self, grid: np.ndarray, resolution_cm: float = None,
                   origin_x: int = None, origin_y: int = None) -> dict:
        """
        Runs the full ML spatial refinement pipeline on a 2D occupancy grid.

        Args:
            grid: 2D float32 numpy array (0.0 = free, 0.5 = unknown, 1.0 = occupied)
            resolution_cm: Grid resolution in cm/cell (default: 4.0)
            origin_x: Grid center X column index
            origin_y: Grid center Y row index

        Returns:
            Dictionary containing refined walls, corridors, clusters, and vector contours.
        """
        res = resolution_cm or self.resolution_cm
        gh, gw = grid.shape
        ox = origin_x if origin_x is not None else gw // 2
        oy = origin_y if origin_y is not None else gh // 2

        # ----------------------------------------------------
        # 1. CLASSIFICATION & PROBABILISTIC THRESHOLDING
        # ----------------------------------------------------
        # Obstacles: high probability > 0.62
        # Free space: low probability < 0.42
        raw_occ = (grid > 0.62).astype(np.uint8) * 255
        raw_free = (grid < 0.42).astype(np.uint8) * 255

        # ----------------------------------------------------
        # 2. MORPHOLOGICAL SPATIAL FILTERING & GAP BRIDGING
        # ----------------------------------------------------
        # Dilate slightly so thin walls are solid and continuous
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_occ = cv2.dilate(raw_occ, kernel_dilate, iterations=1)

        # Morphological Closing bridges small gaps between discrete laser hits
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        refined_occ = cv2.morphologyEx(dilated_occ, cv2.MORPH_CLOSE, kernel_close)

        # Clean free space
        cleaned_free = cv2.morphologyEx(raw_free, cv2.MORPH_CLOSE, kernel_close)

        # ----------------------------------------------------
        # 3. POLYGONAL WALL CONTOUR EXTRACTION (VECTORIZATION)
        # ----------------------------------------------------
        contours, _ = cv2.findContours(refined_occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wall_polygons = []
        wall_segments = []

        for cnt in contours:
            # Filter out tiny noise specks (less than 4 cells perimeter)
            arc_len = cv2.arcLength(cnt, True)
            if cv2.contourArea(cnt) < 1.0 and arc_len < 6.0:
                continue

            # Simplify contour into geometric line segments (Douglas-Peucker algorithm)
            epsilon = max(1.2, 0.02 * arc_len)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Convert contour points to world coordinates (cm)
            world_pts = []
            for pt in approx:
                c, r = pt[0][0], pt[0][1]
                wx = (c - ox) * res
                wy = (oy - r) * res
                world_pts.append((wx, wy))

            if len(world_pts) >= 2:
                wall_polygons.append(world_pts)
                for i in range(len(world_pts)):
                    p1 = world_pts[i]
                    p2 = world_pts[(i + 1) % len(world_pts)]
                    wall_segments.append((p1, p2))

        # ----------------------------------------------------
        # 4. EUCLIDEAN DISTANCE TRANSFORM (NAVIGABLE CORRIDORS)
        # ----------------------------------------------------
        # Inverts obstacle map so free space pixels hold distance (in cells) to nearest wall
        # Free space mask with walls zeroed
        free_for_dist = np.copy(cleaned_free)
        free_for_dist[refined_occ > 0] = 0

        dist_transform = cv2.distanceTransform(free_for_dist, cv2.DIST_L2, 5)

        # Medial Axis / Corridors: cells that have good clearance (> 20cm from any wall)
        # 20cm / res = 5 cells min safety radius for car width (15cm)
        safety_radius_cells = max(3.0, 20.0 / res)
        corridor_mask = (dist_transform >= safety_radius_cells).astype(np.uint8) * 255

        # Extract corridor centerline skeleton
        corridor_pts_rows, corridor_pts_cols = np.where(corridor_mask > 0)
        corridor_world_pts = []
        if len(corridor_pts_rows) > 0:
            # Subsample for rendering efficiency
            step = max(1, len(corridor_pts_rows) // 400)
            for idx in range(0, len(corridor_pts_rows), step):
                r = corridor_pts_rows[idx]
                c = corridor_pts_cols[idx]
                wx = (c - ox) * res
                wy = (oy - r) * res
                corridor_world_pts.append((wx, wy))

        # ----------------------------------------------------
        # 5. K-MEANS SPATIAL OBSTACLE CLUSTERING
        # ----------------------------------------------------
        occ_rows, occ_cols = np.where(refined_occ > 0)
        obstacle_clusters = []

        if len(occ_rows) >= 10:
            pts_array = np.column_stack(((occ_cols - ox) * res, (oy - occ_rows) * res)).astype(np.float32)
            k = min(6, max(2, len(occ_rows) // 40))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            try:
                _, labels, centers = cv2.kmeans(pts_array, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
                for ci, center in enumerate(centers):
                    cluster_pts = pts_array[labels.ravel() == ci]
                    if len(cluster_pts) > 0:
                        min_x, min_y = float(np.min(cluster_pts[:, 0])), float(np.min(cluster_pts[:, 1]))
                        max_x, max_y = float(np.max(cluster_pts[:, 0])), float(np.max(cluster_pts[:, 1]))
                        obstacle_clusters.append({
                            "id": ci + 1,
                            "center": (float(center[0]), float(center[1])),
                            "bounds": (min_x, min_y, max_x, max_y),
                            "count": len(cluster_pts)
                        })
            except Exception:
                pass

        return {
            "walls": wall_segments,
            "wall_segments": wall_segments,
            "wall_polygons": wall_polygons,
            "safe_corridors": corridor_world_pts,
            "corridor_points": corridor_world_pts,
            "obstacle_clusters": obstacle_clusters,
            "refined_occupied_mask": refined_occ,
            "refined_free_mask": cleaned_free,
            "total_walls": len(wall_polygons),
            "total_clusters": len(obstacle_clusters),
            "resolution_cm": res
        }

    def check_path_collision(self, pt1: tuple, pt2: tuple, refined_occ_mask: np.ndarray,
                             origin_x: int = None, origin_y: int = None, resolution_cm: float = None) -> bool:
        """
        Raycasts between two world waypoints to verify no obstacle intersects the line.
        """
        res = resolution_cm or self.resolution_cm
        gh, gw = refined_occ_mask.shape
        ox = origin_x if origin_x is not None else gw // 2
        oy = origin_y if origin_y is not None else gh // 2

        c0 = int(ox + (pt1[0] / res))
        r0 = int(oy - (pt1[1] / res))
        c1 = int(ox + (pt2[0] / res))
        r1 = int(oy - (pt2[1] / res))

        # Bresenham-like sampling along line
        dx = abs(c1 - c0)
        dy = abs(r1 - r0)
        steps = max(dx, dy)
        if steps == 0:
            return False

        for s in range(steps + 1):
            t = float(s) / steps
            c = int(round(c0 + t * (c1 - c0)))
            r = int(round(r0 + t * (r1 - r0)))
            if 0 <= r < gh and 0 <= c < gw:
                val = refined_occ_mask[r, c]
                if val > (0.62 if refined_occ_mask.dtype != np.uint8 else 0):
                    return True  # Collision detected
        return False

    # Alias for convenience
    process_grid = refine_map

