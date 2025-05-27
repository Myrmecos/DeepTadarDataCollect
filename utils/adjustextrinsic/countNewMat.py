import yaml
import numpy as np
import cv2
from imagelidaraligner import ImageLidarAligner, readPcd, array_to_pointcloud, visualize_points_by_distance1, visualize_point_clouds

def get_camera_intrinsic_distortion_extrinsic(yaml_file_name):
    with open(yaml_file_name, 'r') as file:
        contents = yaml.safe_load(file)

    IM = np.matrix(contents['camera']["camera_matrix"]).reshape((3, 3))
    distort = np.matrix(contents['camera']["dist_coeffs"])
    EM = np.matrix(contents['camera']['ex_matrix']).reshape((4, 4))

    return IM, distort, EM

def print_matrix(mat):
    x, y = mat.shape
    cnt=0
    for row in range(x):
        if (cnt==1):
            print("\t", end="")
        else:
            cnt=1
        for col in range(y):
            print(mat[row, col], end=",")
        print("")

CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/livox_camera_calib/config/calib.yaml"
im, distort, em = get_camera_intrinsic_distortion_extrinsic(CAMERA_PARAM_PATH)
print(em)

def rotate(x, y, z, em):
    # Extract rotation and translation
    R = em[:3, :3]
    T = em[:3, 3]

    # Define yaw rotation (5° CCW)
    theta_z = np.radians(z)  # Convert degrees to radians. neg: rotate anticlockwise
    R_yaw = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    theta_x = np.radians(x) # neg: rotate down
    R_pitch = np.array([
        [1,           0,              0],
        [0,  np.cos(theta_x), -np.sin(theta_x)],
        [0,  np.sin(theta_x),  np.cos(theta_x)]
    ])

    theta_y = np.radians(y) # neg: rotate left
    R_roll = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [             0, 1,             0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])


    # Apply yaw rotation
    R_new = R_yaw @ R
    R_new = R_roll @ R_new 
    R_new = R_pitch @ R_new

    new_ex_matrix = np.eye(4)
    new_ex_matrix[:3, :3] = R_new
    new_ex_matrix[:3, 3] = T.reshape(-1)
    return new_ex_matrix


# Construct new extrinsic matrix

# new_ex_matrix = rotate(0, 0, 0, em)

# print("New Extrinsic Matrix:====")
# print_matrix(new_ex_matrix)




if __name__=="__main__":
    CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/livox_camera_calib/config/calib.yaml"
    im, distort, em = get_camera_intrinsic_distortion_extrinsic(CAMERA_PARAM_PATH)

    # read image
    image = cv2.imread("test.jpg")

    # read point cloud
    pcd=readPcd("test.pcd")

    while (1):
        ila = ImageLidarAligner(em, im)
        print("current extrinsic matrix: ")
        print_matrix(em)

        # transform
        pts = points_3d = np.asarray(pcd.points, dtype=float)
        pts_2d, pts_3d = ila._project_points_to_image(pts)

        # visualize result
        valid_pts = array_to_pointcloud(pts_3d)
        visualize_points_by_distance1(pts_2d, pts_3d, im, image, [])

        xyz = input("input x y z: ")
        x, y, z = xyz.split(" ")
        x, y, z = int(x), int(y), int(z)
        em = rotate(x, y, z, em)