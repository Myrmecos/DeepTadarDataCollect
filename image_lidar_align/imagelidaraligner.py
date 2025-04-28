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

def visualize_point_clouds(closest_pts, pointcloud):
    # Set colors: red for closest_pts, blue for pointcloud
    closest_pts.paint_uniform_color([1, 0, 0])  # Red
    pointcloud.paint_uniform_color([0, 0, 1])   # Blue
    
    # Visualize both point clouds
    o3d.visualization.draw_geometries([closest_pts, pointcloud])


camera_matrix = [1364.45, 0.0,      958.327,
                0.0,     1366.46,  535.074,
                0.0,     0.0,      1.0     ]
dist_coeffs = [0.0958277, -0.198233, -0.000147133, -0.000430056, 0.000000]

class ImageLidarAligner:
    def __init__(self, extrinsicMatrix, cameraMatrix):
        self.extrinsicMatrix = extrinsicMatrix
        self.cameraMatrix = cameraMatrix
        

    '''
    @param position: the (x, y) pixel coordinate in the image
    @return correspondingPts: points near the pixel
    '''
    def reportPoints(self, image_coord, points_3d):
        # Convert inputs
        image_coord = np.array(image_coord, dtype=float)  # e.g., [u, v]
        image_coord = self._normalize_image_coord(image_coord[0], image_coord[1])

        points_3d = np.asarray(points_3d.points, dtype=float)

        # Project points to image
        points_2d, points_3d = self._project_points_to_image(points_3d)

        #visualize_points_by_distance(points_2d, points_3d)
        
        # Find closest point
        closest_points, _ = self._find_closest_point(image_coord, points_2d, points_3d)
        
        closest_points = self.clearOutliers(closest_points)
        distance = self._average_distance_from_origin(closest_points)
        return closest_points, points_3d, distance

    '''
    Given image pixel coordinate, return its normalized coordinate
    '''
    def _normalize_image_coord(self, x, y):

        fx, fy = self.cameraMatrix[0, 0], self.cameraMatrix[1, 1]  # 1364.45, 1366.46
        cx, cy = self.cameraMatrix[0, 2], self.cameraMatrix[1, 2]  # 958.327, 535.074
        
        # Remove principal point offset and normalize by focal length
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        
        return np.array([x_norm, y_norm])

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

    '''
    Given image coordinate (normalized), pointcloud projected to image and pointcloud,
    return the points in pointcloud that are nearest to the pixel
    '''
    def _find_closest_point(self, image_coord, valid_points_2d, valid_points_3d, num_of_pts=500):
        # Compute Euclidean distances in image plane
        distances = np.linalg.norm(valid_points_2d - image_coord, axis=1)

        # Find indices of the 5000 closest points
        num_points = min(num_of_pts, len(distances))  # Handle cases with fewer than 5000 points
        closest_indices = np.argsort(distances)[:num_points]
        
        return valid_points_3d[closest_indices], distances
        
    '''
    @param pts: a list of points
    @param percentile: the upper and lower percentiles to filter out. for example, percentile=0.25 indicates we select 25% to 75% points and clear other outliers.
    @return clearedPts: a list of points that had outliers removed
    '''
    def clearOutliers(self, points, percentile=25,k=1.5):
        if points.shape[0] == 0:
            return points, np.array([], dtype=bool)
        
        # Calculate quartiles and IQR for each dimension
        q1 = np.percentile(points, percentile, axis=0)
        q3 = np.percentile(points, 100-percentile, axis=0)
        iqr = q3 - q1
        
        # Define lower and upper bounds for each dimension
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        # Create a mask for points within bounds in all dimensions
        mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
        
        # Apply the mask to filter points
        filtered_points = points[mask]
        
        return filtered_points
    
    def _average_distance_from_origin(self, pts):
        """
        Calculate the average Euclidean distance of points from the origin (0,0,0).
        
        Parameters:
        - pts: numpy array of shape (n, 3) containing 3D points
        
        Returns:
        - average distance as a float
        """
        if len(pts) == 0:
            return 0.0
        
        # Calculate Euclidean distances from origin for each point
        distances = np.sqrt(np.sum(pts**2, axis=1))
        
        # Return the average distance
        return np.mean(distances)
        


if __name__=="__main__":
    # read extrinsic param
    extrinsic = readExtrinsic("/home/astar/dart_ws/calib/extrinsic.txt")

    # read image
    image = cv2.imread("/home/astar/dart_ws/single_scene_calibration/0.png")
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()

    # read point cloud
    pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
    #o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", width=800, height=600)

    # get coordinates
    coord = [645.952, 677.306]

    cameraMatrix = np.array(camera_matrix).reshape(3, 3)
    ila = ImageLidarAligner(extrinsic, cameraMatrix)

    # transform
    closest_pts, valid_pts, dist = ila.reportPoints(coord, pcd)
    print("average distance from origin is:", dist)
    print(closest_pts[0, :])

    # visualize result
    closest_pts = array_to_pointcloud(closest_pts)
    valid_pts = array_to_pointcloud(valid_pts)

    visualize_point_clouds(valid_pts, closest_pts)

    print(closest_pts, dist)
