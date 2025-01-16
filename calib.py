import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
import mplcursors
import os

#
#根据需求，认为需要把distance对应到transform上，即将distance通过resize，平移和旋转贴合到transform上。
#下方示例是将transform通过转换对应到distance上，只要把transform和distance的文件交换即可。
#
cursor1 = None
cursor2 = None

# Load the YAML file
def load_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    # Extract points
    transform_points = config['transform_points']
    reference_points = config['reference_points']

    return (transform_points, reference_points)

# transform image give the rotation matrix R, translation matrix T and scale s
def transform_img(transform_image, R, T, s):
    # Load the reference image
    # transform = np.load("color.npy")
    
    # Define the padding size (top, bottom, left, right)
    print("image shape:", transform_image.shape)
    if (transform_image.shape[1]-transform_image.shape[0]>0):
        top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
        bottom = abs(transform_image.shape[1]-transform_image.shape[0])
        left = 0
        right = 0
    else: 
        # we will check later how to deal with a tall image
        print("it's a tall image")
        #exit(1)
        top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
        bottom = 0#abs(transform_image.shape[1]-transform_image.shape[0])
        left = 0
        right = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
    #Zero-pad the image using cv2.copyMakeBorder
    transform_image = cv.copyMakeBorder(transform_image, top, bottom, left, right, cv.BORDER_CONSTANT, value=np.nan)

    #transform_image = cv.rotate(transform_image, cv.ROTATE_90_CLOCKWISE) #testing only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #transform= cv.cvtColor(transform_image, cv.COLOR_BGR2GRAY) #testing only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Get the shape of the reference image
    transform = transform_image
    height, width = transform.shape

    # Create a grid of coordinates for the reference image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.vstack((x.flatten(), y.flatten()))  # Shape: (2, N), where N = height * width

    # Step 1: Apply scaling
    coords_scaled = coords

    # Step 2: Apply rotation and translation
    coords_transformed = np.dot(R, coords_scaled) + T[:, np.newaxis]
    print("after rotation: ", coords_transformed)

    # Reshape the transformed coordinates back to the image shape
    x_transformed = coords_transformed[0, :].reshape(height, width)
    y_transformed = coords_transformed[1, :].reshape(height, width)

    print("transform ok")

    # Resize the distance image using interpolation
    #resized_distance = cv.resize(transformed_reference, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    transformed_reference = map_coordinates(transform_image, [y_transformed, x_transformed], order=1, mode='constant', cval=np.nan)
    transformed_reference = cv.resize(transformed_reference, (int(transform.shape[0]/s), int(transform.shape[1]/s)), interpolation=cv.INTER_AREA)
    print("resize ok")

    return transformed_reference

#calculates rotation matrix R, transformation matrix T and Scaling factor s
def calc_RT(reference_points, transform_points):
     # Step 1: Center the points
    transform_centroid = np.mean(transform_points, axis=0)
    reference_centroid = np.mean(reference_points, axis=0)

    transform_centered = transform_points - transform_centroid
    reference_centered = reference_points - reference_centroid


    # Step 2: Compute the covariance matrix
    H = reference_centered.T @ transform_centered

    # Step 3: Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Step 4: Compute the rotation matrix R
    R = Vt.T @ U.T

    # Handle reflection case (ensure determinant of R is 1)
    if np.linalg.det(R) < 0:
        print("determinant of R is negative, flip it.")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 5: Compute the translation vector T
    T = transform_centroid - R @ reference_centroid

    # Print the results
    print("Rotation Matrix (R):")
    print(R)
    print("Translation Vector (T):")
    print(T)
    return (R, T)

# calculate scale give reference (thermal) and transform (depth) points
def calc_scale(reference_points, transform_points):
    P_d = np.array(reference_points)  # reference image points
    P_t = np.array(transform_points)  # transform image points
    # Compute scaling factor
    dist_d = 0
    dist_t = 0
    for i in range(1, len(P_d)):
        dist_d += np.linalg.norm(P_d[i] - P_d[0])  # Distance between two points in reference image
        dist_t += np.linalg.norm(P_t[i] - P_t[0])  # Distance between two points in transform image
    
    s = dist_t / dist_d  # Scaling factor
    return s

def on_mouse_move(event):
    if event.inaxes == ax1:
        cursor2.set_data(event.xdata, event.ydata)
    elif event.inaxes == ax2:
        cursor1.set_data(event.xdata, event.ydata)
    plt.draw()


if __name__=="__main__":
    transform_points, reference_points = load_yaml("config.yaml")
    baseDir = "RawData/exp04/"
    transform_dir = "realsense_depth/"
    reference_dir = "MLX/"
    ind = 3
    transform_files = os.listdir(baseDir+transform_dir)
    reference_files = os.listdir(baseDir+reference_dir)
    

    # calculate scale
    s = calc_scale(reference_points, transform_points)
    print("scaling factor (transform/reference) is:", s)

    # rescale refernce points
    reference_points = np.array(reference_points)*s

    # calculate rotation and transformation
    R, T = calc_RT(reference_points, transform_points)

    # map the transform array to reference image
    transform_image=np.load(baseDir+"realsense_depth/"+transform_files[ind])
    #transform_image = np.load("ori.npy")
    #transform_image = cv.imread("shifted.png")
    transform_image= cv.cvtColor(transform_image, cv.COLOR_BGR2GRAY)
    transform_image = cv.normalize(transform_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    reference_image = np.load(baseDir+"seek_thermal/"+reference_files[ind])
    #reference_image = np.load("trans.npy")
    #reference_image = cv.imread("shifted.png")
    reference_image= cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)
    reference_image = cv.normalize(reference_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    transform_image = transform_img(transform_image, R, T, s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(transform_image, cmap='gray')
    ax1.set_title('transform Image')
    ax2.imshow(reference_image, cmap='gray')
    ax2.set_title('reference Image')

    #add cursor
    cursor1, = ax1.plot([], [], 'r+')
    cursor2, = ax2.plot([], [], 'r+')
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    plt.tight_layout()
    plt.title("Transformed reference Image")
    plt.show()

    
