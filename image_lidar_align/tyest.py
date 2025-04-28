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

class ImageLidarAligner:
    def __init__(self, extrinsicMatrix, image_shape):
        self.extrinsicMatrix = extrinsicMatrix
        self.image_shape = image_shape

    '''
    @param position: the (x, y) pixel coordinate in the image
    @return correspondingPts: points near the pixel
    '''
    def reportPoints(self, image_coord, points_3d):
        points_3d = np.asarray(points_3d.points, dtype=float)
        points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # [x, y, z, 1]

        print(points_3d.shape)

        # Transform points to camera coordinates
        points_camera = self.extrinsicMatrix @ points_3d.T  # Apply extrinsic matrix
        points_camera = points_camera[:3, :]  # Discard homogeneous coordinate

        # Filter points with z > 0 (in front of camera)
        valid = points_camera[2, :] > 0
        points_camera = points_camera[:, valid]

        # Project to 2D image plane
        points_2d = points_camera

        # Visualize
        plt.figure(figsize=(6, 4))
        plt.scatter(points_2d[0, :], points_2d[1, :], s=1, c='blue')
        plt.gca().invert_yaxis()  # Image coordinates: y-axis points down
        plt.title('Projected 2D Point Cloud')
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.savefig('point_cloud_projection.png')
        plt.close()

    '''
    projects points to 2d
    valid: points that are in front of camera
    return: points_2d: indices of valid points converted to 2d
    '''
    def _project_points_to_image(self, points_3d):
        points_3d_homog = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_camera = points_3d_homog @ self.extrinsicMatrix.T
        valid_mask = points_camera[:, 2] > 0
        valid_indices = np.where(valid_mask)[0]
        points_camera_valid = points_camera[valid_mask]
        f, cx, cy = 1.0, 0.0, 0.0  # Assume defaults
        points_2d = np.zeros((len(valid_indices), 2))
        points_2d[:, 0] = f * (points_camera_valid[:, 0] / points_camera_valid[:, 2]) + cx
        points_2d[:, 1] = f * (points_camera_valid[:, 1] / points_camera_valid[:, 2]) + cy
        return points_2d, valid_indices

    def _find_closest_point(self, image_coord, points_2d, points_3d, valid):
        # Filter valid projected points
        valid_points_2d = points_2d[valid]
        valid_points_3d = points_3d[valid]
        
        # Compute Euclidean distances in image plane
        distances = np.linalg.norm(valid_points_2d - image_coord, axis=1)

        # Find indices of the 5000 closest points
        num_points = min(500000, len(distances))  # Handle cases with fewer than 5000 points
        closest_indices = np.argsort(distances)[:num_points]
        
        return valid_points_3d[closest_indices], distances[closest_indices]
        
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
    plt.imshow(image)
    
    #plt.show()

    pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
    #o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", width=800, height=600)
    coord = [1176, 499]

    ila = ImageLidarAligner(extrinsic, image_shape=image.shape)
    ila.reportPoints(coord, pcd)

    # pointcloud = array_to_pointcloud(closest_pts)
    # #o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud Visualization", width=800, height=600)

    # visualize_point_clouds(pcd, pointcloud)

    # print(closest_pts, dist)
