# TODO: 
# 1. accepts a point cloud data
# 2. accept an image data
# 3. given the coordinate on the image, report the corresponding points near the position in the point cloud
import open3d as o3d 
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import yaml
import math

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

def get_camera_intrinsic_distortion_extrinsic(yaml_file_name):
        with open(yaml_file_name, 'r') as file:
            contents = yaml.safe_load(file)
    
        IM = np.matrix(contents['camera']["camera_matrix"]).reshape((3, 3))
        distort = np.matrix(contents['camera']["dist_coeffs"])
        EM = np.matrix(contents['camera']['ex_matrix']).reshape((4, 4))

        return IM, distort, EM

def get_cam_rotor_matrix(yaml_file_name):
    with open(yaml_file_name, 'r') as file:
        contents = yaml.safe_load(file)
    return np.asarray(contents["cam_rotor"]).reshape((4, 4))
        
def visualize_points_by_distance(points_2d, points_3d, target_pts=[]):
    points_2d=points_2d[:600000,]
    points_3d = points_3d[:600000,]
    # Calculate Euclidean distance from origin for each 3D point
    distances = np.linalg.norm(points_3d, axis=1)
    
    # Create scatter plot with colors based on distance
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=distances, s=1, cmap='viridis')
    
    if len(target_pts)!=0:
        plt.scatter(target_pts[:, 0], target_pts[:, 1], c="red", s=5)
    
    plt.colorbar(scatter, label='3D Distance from Origin')
    plt.gca().invert_yaxis()  # Image coordinates: y-axis points down
    plt.title('2D Points Colored by 3D Distance')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-0.3, 0.3])
    plt.ylim([0.3, -0.3])
    
    plt.show()

def visualize_point_clouds(closest_pts, pointcloud):
    # Set colors: red for closest_pts, blue for pointcloud
    closest_pts.paint_uniform_color([1, 0, 0])  # Red
    pointcloud.paint_uniform_color([0, 0, 1])   # Blue
    
    # Visualize both point clouds
    o3d.visualization.draw_geometries([closest_pts, pointcloud])

def visualize_points_by_distance1(points_2d, points_3d, cameraMatrix, image, target_pts=[]):
    # Limit to 600,000 points for performance
    points_2d = points_2d[:400000]
    points_3d = points_3d[:400000]

    # Extract intrinsic parameters
    fx = cameraMatrix[0, 0]  # Focal length x
    fy = cameraMatrix[1, 1]  # Focal length y
    cx = cameraMatrix[0, 2]  # Principal point x
    cy = cameraMatrix[1, 2]  # Principal point y

    # Convert normalized coordinates to pixel coordinates
    points_2d_pixel = np.zeros_like(points_2d)
    points_2d_pixel[:, 0] = points_2d[:, 0] * fx + cx  # x_pixel = x_norm * fx + cx
    points_2d_pixel[:, 1] = points_2d[:, 1] * fy + cy  # y_pixel = y_norm * fy + cy

    # Convert target points to pixel coordinates (if provided)
    if len(target_pts) != 0:
        target_pts_pixel = np.zeros_like(target_pts)
        target_pts_pixel[:, 0] = target_pts[:, 0] * fx + cx
        target_pts_pixel[:, 1] = target_pts[:, 1] * fy + cy
    else:
        target_pts_pixel = []

    # Calculate Euclidean distance from origin for each 3D point
    distances = np.linalg.norm(points_3d, axis=1)

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.imshow(image, alpha=0.5)  # Display the image

    # Plot points, colored by 3D distance
    scatter = plt.scatter(
        points_2d_pixel[:, 0], points_2d_pixel[:, 1],
        c=distances, s=1, cmap='viridis', alpha=0.9
    )

    # Plot target points (if provided)
    if len(target_pts_pixel) != 0:
        plt.scatter(
            target_pts_pixel[:, 0], target_pts_pixel[:, 1],
            c="red", s=5, label='Target Points', alpha=0.6
        )

    # Add colorbar
    plt.colorbar(scatter, label='3D Distance from Origin')

    image_size = image.shape
    # Set axes limits to match image dimensions
    plt.xlim(0, image_size[1])
    plt.ylim(image_size[0], 0)  # Invert y-axis to match image coordinates
    plt.title('Point Cloud on Original Image')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.gca().set_aspect('equal')  # Maintain aspect ratio
    if len(target_pts_pixel) != 0:
        plt.legend()

    plt.show()


camera_matrix = [1364.45, 0.0,      958.327,
                0.0,     1366.46,  535.074,
                0.0,     0.0,      1.0     ]
dist_coeffs = [0.0958277, -0.198233, -0.000147133, -0.000430056, 0.000000]

class ImageLidarAligner:
    def __init__(self, extrinsicMatrix, cameraMatrix, num_of_points = 30):
        self.extrinsicMatrix = extrinsicMatrix
        self.cameraMatrix = cameraMatrix
        self.num_of_points = num_of_points
        

    '''
    @param position: the (x, y) pixel coordinate in the image
    @return correspondingPts: points near the pixel
    '''
    def reportPoints(self, image_coord, points_3d):
        # Convert inputs
        image_coord = np.array(image_coord, dtype=float)  # e.g., [u, v]
        image_coord = self._normalize_image_coord(image_coord[0], image_coord[1])

        points_3d = np.asarray(points_3d.points, dtype=float)

        print(points_3d.shape)
        
        # Project points to image
        points_2d, points_3d = self._project_points_to_image(points_3d)

        #visualize_points_by_distance(points_2d, points_3d)
        
        # Find closest point
        closest_points, _ = self._find_closest_point(image_coord, points_2d, points_3d, num_of_pts=self.num_of_points)
        
        closest_points = self.clearOutliers(closest_points)
        distance = self._average_distance_from_origin(closest_points)

        return closest_points, points_3d, distance

        '''
    @param position: the (x, y) pixel coordinate in the image
    @return correspondingPts: points near the pixel
    '''
    def reportPoints1(self, image_coord, points_3d):
        # Convert inputs
        image_coord = np.array(image_coord, dtype=float)  # e.g., [u, v]
        image_coord = self._normalize_image_coord(image_coord[0], image_coord[1])

        points_3d = np.asarray(points_3d.points, dtype=float)

        print(points_3d.shape)
        
        # Project points to image
        points_2d, points_3d = self._project_points_to_image(points_3d)

        #visualize_points_by_distance(points_2d, points_3d)
        
        # Find closest point
        closest_points, _ = self._find_closest_point(image_coord, points_2d, points_3d, num_of_pts=self.num_of_points)
        
        closest_points = self.clearOutliers(closest_points)
        distance = self._average_distance_from_origin(closest_points)

        return closest_points, points_2d, points_3d, distance

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
        points_camera = np.asarray(points_camera)
        valid_mask = (points_camera[:, 2] > 0).flatten()
        valid_mask = valid_mask.flatten()
        print("DEBUG: ", points_camera.shape, valid_mask.shape)
    
        points_2d_valid = points_camera[valid_mask]

        points_3d_valid = points_3d[valid_mask]

        points_2d_valid = points_2d_valid[:, :2] / points_2d_valid[:, 2][:, np.newaxis]

        return points_2d_valid, points_3d_valid

    '''
    Given image coordinate (normalized), pointcloud projected to image and pointcloud,
    return the points in pointcloud that are nearest to the pixel
    '''
    def _find_closest_point(self, image_coord, valid_points_2d, valid_points_3d, num_of_pts=50):
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
    
    '''transform points from lidar coord to rotor coord'''
    def to_rotor_coord(self, pts, rotor_cam_em):
        pts = np.array(pts)
        points_3d_homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_camera = self.extrinsicMatrix @ points_3d_homog.T
        #print(pts_camera.shape, "============+++++++++++++++++++")
        pts_rotor = rotor_cam_em @ pts_camera
        #print("camera: ", pts_camera[0, :])
        return pts_rotor.T
    
    '''transform points at rotor coord to yaw (pos direction: clockwise)'''
    def to_degree(self, rotor_coords):
        centerR = np.mean(rotor_coords, axis = 0)
        arctanval = centerR[0,0]/centerR[0,2]
        return math.atan(arctanval)/(math.pi)*180

    def calc_yaw(self, pts, rotor_cam_em):
        rotor_coords = self.to_rotor_coord(pts, rotor_cam_em)
        return self.to_degree(rotor_coords)



if __name__=="__main__":
    # read extrinsic param
    #extrinsic = readExtrinsic("/home/astar/dart_ws/calib/extrinsic.txt")
    CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/livox_camera_calib/config/calib.yaml"
    im, distort, em = get_camera_intrinsic_distortion_extrinsic(CAMERA_PARAM_PATH)
    cam_rotor_em = get_cam_rotor_matrix(CAMERA_PARAM_PATH)
    # read image
    #image = cv2.imread("/home/astar/dart_ws/single_scene_calibration/0.png")
    #image = cv2.imread("/home/astar/dart_ws/calib/calibimage/test4.jpg")
    #image = cv2.imread("testImg/frame0000.jpg")
    image = cv2.imread("test.jpg")
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    #visualize_normalized_image(image, im)

    # read point cloud
    #pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
    #pcd = readPcd("/home/astar/dart_ws/calib/calibpointcloud/calibscene_test_cropped.pcd")
    #pcd = readPcd("/home/astar/dart_ws/testing_data/test2.pcd")
    pcd=readPcd("test.pcd")
    #o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", width=800, height=600)

    # get coordinates
    #coord = [1334, 1187]
    # coord=[2191, 1631]
    coord = [100, 1631]

    #inp = input("input coordinate of green light: ")
    inp = ""
    if len(inp)==0:
        coord = [2000, 2000]
    else:
        inp = inp.split(" ")
        coord = [int(inp[0]), int(inp[1])]

    cameraMatrix = np.array(camera_matrix).reshape(3, 3)
    # ila = ImageLidarAligner(extrinsic, cameraMatrix)
    ila = ImageLidarAligner(em, im)
    

    # transform
    closest_pts, pts_2d, pts_3d, dist = ila.reportPoints1(coord, pcd)
    target_pts, _ = ila._project_points_to_image(closest_pts)
    print("average distance from origin is:", dist)
    print(closest_pts[0, :])

    # visualize result
    closest_pts_pcd = array_to_pointcloud(closest_pts)
    valid_pts = array_to_pointcloud(pts_3d)
    #visualize_points_by_distance1(pts_2d, pts_3d, im, image, target_pts)
    #visualize_point_clouds(valid_pts, closest_pts_pcd)

    centerC = np.mean(closest_pts, axis = 0) 
    #print("center for light at camera coord: ", centerC)

    #positive direction: 
    #  
    #   for lidar, x is front, y is left, z is up
    # for camera, x is right, y is down, z is front
    rotor_coords = ila.to_rotor_coord(closest_pts, cam_rotor_em).T
    print("lidar: ", closest_pts[0])
    print("rotor: ", rotor_coords[0])
    centerR = np.mean(rotor_coords, axis = 0)
    print("center for light at rotor coord: ", centerR)
    arctanval = centerR[0,0]/centerR[0,2]
    print("arctan val of angle: ", arctanval)
    print("degrees: ", math.atan(arctanval)*180/(math.pi))
    print(ila.to_degree(rotor_coords))


    # print(closest_pts, dist)
