# TODO: 
# 1. accepts a point cloud data
# 2. accept an image data
# 3. given the coordinate on the image, report the corresponding points near the position in the point cloud
import open3d as o3d 
import numpy as np 
import matplotlib.pyplot as plt
import cv2

def readPcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("readPcd received empty point cloud")
        return
    return pcd

def readExtrinsic(path):
    output = np.genfromtxt(path, delimiter=',', dtype=float)
    return output.reshape(4, 4)

def array_to_pointcloud(points_array):
    # Ensure points_array is Nx3
    points_array = np.asarray(points_array, dtype=float).reshape(-1, 3)
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    # Set points
    pcd.points = o3d.utility.Vector3dVector(points_array)
    return pcd

def visualize_points_by_distance(points_2d, points_3d):
    points_2d=points_2d[:60000,]
    points_3d = points_3d[:60000,]
    # Calculate Euclidean distance from origin for each 3D point
    distances = np.linalg.norm(points_3d, axis=1)
    
    # Create scatter plot with colors based on distance
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=distances, s=1, cmap='viridis')
    plt.colorbar(scatter, label='3D Distance from Origin')
    plt.gca().invert_yaxis()  # Image coordinates: y-axis points down
    plt.title('2D Points Colored by 3D Distance')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()

def normalize_image_coord(x, y, camera_matrix):
    camera_matrix = np.array(camera_matrix).reshape(3, 3)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]  # 1364.45, 1366.46
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]  # 958.327, 535.074
    
    # Remove principal point offset and normalize by focal length
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    
    return np.array([x_norm, y_norm])

camera_matrix = [1364.45, 0.0,      958.327,
                0.0,     1366.46,  535.074,
                0.0,     0.0,      1.0     ]
dist_coeffs = [0.0958277, -0.198233, -0.000147133, -0.000430056, 0.000000]

class ImageLidarAligner:
    def __init__(self, extrinsicMatrix, image_shape):
        self.extrinsicMatrix = extrinsicMatrix
        self.image_shape = np.array(image_shape)[:2]

    '''
    @param position: the (x, y) pixel coordinate in the image
    @return correspondingPts: points near the pixel
    '''
    def reportPoints(self, image_coord, points_3d):
        # Convert inputs
        image_coord = np.array(image_coord, dtype=float)  # e.g., [u, v]
        image_coord = normalize_image_coord(image_coord[0], image_coord[1], camera_matrix)
        print("normalized image coord: ", image_coord)
        points_3d = np.asarray(points_3d.points, dtype=float)

        # Project points to image
        points_2d, points_3d = self._project_points_to_image(points_3d)

        visualize_points_by_distance(points_2d, points_3d)
        print(max(points_2d[:, 0]))
        # image coordinate normalization
        print("shape of points_2d:", points_2d.shape)
        #image_coord = (image_coord-(self.image_shape/2))/(self.image_shape/2)
        # Find closest point
        closest_point, distance = self._find_closest_point(image_coord, points_2d, points_3d)
        
        return closest_point, points_3d, distance

    '''
    projects points to 2d
    valid: points that are in front of camera
    return: points_2d: indices of valid points converted to 2d
    '''
    def _project_points_to_image(self, points_3d):
        points_3d_homog = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_camera = self.extrinsicMatrix @ points_3d_homog.T
        points_camera = points_camera[:3, :].T

        valid_mask = points_camera[:, 2] > 0

        points_2d_valid = points_camera[valid_mask]

        points_3d_valid = points_3d[valid_mask]

        points_2d_valid = points_2d_valid[:, :2] / points_2d_valid[:, 2][:, np.newaxis]

        return points_2d_valid, points_3d_valid

    def _find_closest_point(self, image_coord, valid_points_2d, valid_points_3d):
        
        print("coordinate in image: ", image_coord)
        print("image shape: ", self.image_shape)
        # Compute Euclidean distances in image plane
        print("valid points shape: ", valid_points_2d.shape)
        distances = np.linalg.norm(valid_points_2d - image_coord, axis=1)

        # Find indices of the 5000 closest points
        num_points = min(500, len(distances))  # Handle cases with fewer than 5000 points
        closest_indices = np.argsort(distances)[:num_points]
        
        return valid_points_3d[closest_indices], distances
        
    '''
    @param pts: a list of points
    @param percentile: the upper and lower percentiles to filter out. for example, percentile=0.25 indicates we select 25% to 75% points and clear other outliers.
    @return clearedPts: a list of points that had outliers removed
    '''
    def clearOutliers(pts, percentile):
        pass


def visualize_point_clouds(closest_pts, pointcloud):
    # Set colors: red for closest_pts, blue for pointcloud
    closest_pts.paint_uniform_color([1, 0, 0])  # Red
    pointcloud.paint_uniform_color([0, 0, 1])   # Blue
    
    # Visualize both point clouds
    o3d.visualization.draw_geometries([closest_pts, pointcloud])

if __name__=="__main__":
    extrinsic = readExtrinsic("/home/astar/dart_ws/calib/extrinsic.txt")

    image = cv2.imread("/home/astar/dart_ws/single_scene_calibration/0.png")
    print(image.shape)
    plt.imshow(image)
    
    plt.show()

    pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
    #o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", width=800, height=600)
    coord = [645.952, 677.306]

    ila = ImageLidarAligner(extrinsic, image_shape=image.shape)
    closest_pts, valid_pts, dist = ila.reportPoints(coord, pcd)

    closest_pts = array_to_pointcloud(closest_pts)
    valid_pts = array_to_pointcloud(valid_pts)
    #o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud Visualization", width=800, height=600)

    visualize_point_clouds(valid_pts, closest_pts)

    print(closest_pts, dist)
