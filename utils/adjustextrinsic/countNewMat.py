import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imagelidaraligner import ImageLidarAligner, readPcd, array_to_pointcloud, visualize_points_by_distance1, visualize_point_clouds

def plot2dpoints(ax, points_2d, points_3d, cameraMatrix, target_pts = []):
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
    scatter = ax.scatter(
        points_2d_pixel[:, 0], points_2d_pixel[:, 1],
        c=distances, s=0.5, cmap='viridis', alpha=0.8
    )

    # Plot target points (if provided)
    if len(target_pts_pixel) != 0:
        ax.scatter(
            target_pts_pixel[:, 0], target_pts_pixel[:, 1],
            c="red", s=5, label='Target Points', alpha=0.3
        )

'''
plot on the axis plt
'''
def interactive_compare(points_2d, points_3d, cameraMatrix, image, target_pts=[]):
    # Limit to 600,000 points for performance
    points_2d = points_2d[:600000]
    points_3d = points_3d[:600000]
    

    # Create figure=================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 14))
    ax1.imshow(image, alpha=0.8)  # Display the image

    plot2dpoints(ax2, points_2d, points_3d, cameraMatrix, target_pts)

    # Add colorbar
    #plt.colorbar(scatter, label='3D Distance from Origin')

    image_size = image.shape
    # Set axes limits to match image dimensions
    
    ax2.set_xlim(0, image_size[1])
    ax2.set_ylim(image_size[0], 0)  # Invert y-axis to match image coordinates
    plt.title('Point Cloud on Original Image')
    
    ax2.set_aspect('equal')  # Maintain aspect ratio
    # if len(target_pts_pixel) != 0:
    #     plt.legend()

    # interactiveness
    cross1, = ax1.plot([], [], '+', color='red', ms=10)  # Red cross on ax1
    cross2, = ax2.plot([], [], '+', color='red', ms=10)  # Red cross on ax2

    def on_motion(event):
        if event.inaxes == ax1:  # Check if cursor is in ax1

            x, y = ax1.transData.inverted().transform((event.x, event.y))
            y -= 15
            # Update cross positions
            cross1.set_data(x, y)
            cross2.set_data(x, y)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    plt.show()





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

CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/lidar_image_align/calib/calib.yaml"
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
    CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/lidar_image_align/calib/calib.yaml"
    im, distort, em = get_camera_intrinsic_distortion_extrinsic(CAMERA_PARAM_PATH)

    # read image
    image = cv2.imread("target/test4.jpg")

    # read point cloud
    pcd=readPcd("target/test4.pcd")

    while (1):
        ila = ImageLidarAligner(em, im)
        print("current extrinsic matrix: ")
        print_matrix(em)

        # transform
        pts = points_3d = np.asarray(pcd.points, dtype=float)
        pts_2d, pts_3d = ila._project_points_to_image(pts)

        # visualize result
        valid_pts = array_to_pointcloud(pts_3d)
        #visualize_points_by_distance1(pts_2d, pts_3d, im, image, [])
        interactive_compare(pts_2d, pts_3d, im, image)

        xyz = input("input x y z: ")
        x, y, z = xyz.split(" ")
        x, y, z = float(x), float(y), float(z)
        em = rotate(x, y, z, em)




# current res: 
# 0.015269788005512014,-0.9995062832982383,-0.027452250353914978,0.0865,
#         -0.019947232337172653,0.027145477890242256,-0.9994322542211479,0.005,
#         0.9996848057130043,0.01580871612067855,-0.01952288159159528,-0.00036764,
#         0.0,0.0,0.0,1.0