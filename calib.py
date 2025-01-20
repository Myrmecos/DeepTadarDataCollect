import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from matplotlib.widgets import Slider
import mplcursors
import os
import math

#
#根据需求，认为需要把distance对应到transform上，即将distance通过resize，平移和旋转贴合到transform上。
#下方示例是将transform通过转换对应到distance上，只要把transform和distance的文件交换即可。
#

# TODO: seek/1m.yaml; 2m.yaml; ... 7m.yaml;
# senxor_m08/...; senxor_m16/...; mlx/...;
cursor1 = None
cursor11 = None
cursor2 = None
cursor21 = None
im = None
transform_image_ori = None
R = None
T = None
s = 0

# Load the YAML file
def load_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    # Extract points
    transform_points = config['transform_points']
    reference_points = config['reference_points']

    return (transform_points, reference_points)

# transform image give the rotation matrix R, translation matrix T and scale s
def transform_img(transform_image, R, T, scale):
    # Load the reference image
    # transform = np.load("color.npy")
    
    # Define the padding size (top, bottom, left, right)
    print("image shape:", transform_image.shape)
    if (transform_image.shape[1]-transform_image.shape[0]>=0):
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
    transformed_reference = cv.resize(transformed_reference, (int(transform.shape[0]/scale), int(transform.shape[1]/scale)), interpolation=cv.INTER_AREA)
    print("resize ok")



    return transformed_reference

# calculates rotation matrix R, transformation matrix T and Scaling factor s
# caution: relative distance of reference points must already be scaled to same scale 
# (i.e. only rotation and translation, no scaling is needed)
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

# callback function, for moving cursor
def on_mouse_move(event):
    if event.inaxes == ax1:
        cursor11.set_data(event.xdata, event.ydata)
        cursor2.set_data(event.xdata, event.ydata)
    elif event.inaxes == ax2:
        cursor21.set_data(event.xdata, event.ydata)
        cursor1.set_data(event.xdata, event.ydata)
    plt.draw()

#draw a black margin inside image without affecting image shape.
def draw_black_margin(image, margin_width):
    height, width = image.shape
    image[:margin_width, :] = 0  # Top margin
    image[-margin_width:, :] = 0  # Bottom margin
    image[:, :margin_width] = 0  # Left margin
    image[:, -margin_width:] = 0  # Right margin
    return image

def update(val):
    global R, T
    # Get current slider values
    angle = angle_slider.val
    xshift = xshift_slider.val
    yshift = yshift_slider.val
    scale = scale_slider.val

    # Apply transformations
    R = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    T = np.array([xshift, yshift])
    print(R.shape, T.shape, type(R))
    print(R, T, scale)
    # Update the displayed image
    layer1 = transform_img(transform_image_ori, R, T, scale)
    layer2 = reference_image
    im.set_data(layer1)
    #res = 0.5*layer1 + 0.5*layer2
    #ax1.clear()
    #ax1.imshow(layer1, alpha=0.5)
    #ax1.imshow(layer2, alpha=0.5)
    fig.canvas.draw_idle()


def MLX_process(MLX_temperature_map):
    MLX_temperature_map = MLX_temperature_map.astype(np.uint8)
    print("MLX: ", MLX_temperature_map)
    MLX_temperature_map = cv.normalize(MLX_temperature_map, None, 0, 255, cv.NORM_MINMAX)
    #MLX_temperature_map = cv.resize(MLX_temperature_map, (320, 240), interpolation=cv.INTER_NEAREST)
    MLX_temperature_map = cv.applyColorMap(MLX_temperature_map, cv.COLORMAP_JET)
    return MLX_temperature_map

def dump_yaml(R, T, s, filename):
    R = R.tolist()
    T = T.tolist()
    s = float(s)
    
    data = {
        "R": R, "T": T, "s": s
    }
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=True)

def read_yaml(filename):
    with open(filename) as file:
        data = yaml.safe_load(file)
    return np.array(data["R"]), np.array(data["T"]), np.float64(data["s"])

if __name__=="__main__":
    margin = 0
    # load image
    src_distance = "4"
    dest_distance = "4" #which distance we want to adjust our RTS to(e.g. we can read calib result at 7m, transform it to use at 6m)
    baseDir = "RawData/exp2"+dest_distance+"/"
    transform_dir = "realsense_depth/"
    reference_dir = "MLX/"
    mode = "adjust" # adjust previous R, T, S
    #mode = "pointcalib"

    
    pointsfile = "calibpoints/"+reference_dir[:-1]+".yaml"
    RTSfileSrc = "calibresults/"+reference_dir+src_distance+".yaml"
    RTSfileDst = "calibresults/"+reference_dir+dest_distance+".yaml"
    #RTSfile = "seekRTS.yaml"
    ind = 1

    transform_points, reference_points = load_yaml(pointsfile)
    transform_files = os.listdir(baseDir+transform_dir)
    reference_files = os.listdir(baseDir+reference_dir)

    if mode != "adjust":
        # calculate scale
        scale = calc_scale(reference_points, transform_points)
        #print("scaling factor (transform/reference) is:", scale)

        # rescale refernce points
        reference_points = np.array(reference_points)*scale

        # calculate rotation and transformation
        R, T = calc_RT(reference_points, transform_points)

    else:
        R, T, scale = read_yaml(RTSfileSrc)
        print("read: ", R, T, scale)

    # map the transform array to reference image
    #1. load the to-be-conferted image and reference image
    transform_image=np.load(baseDir+transform_dir+transform_files[ind])
    transform_image = cv.normalize(transform_image, None, 0, 255, cv.NORM_MINMAX)
    transform_image= cv.cvtColor(transform_image, cv.COLOR_BGR2GRAY)
    # transform_image = cv.normalize(transform_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    reference_image = np.load(baseDir+reference_dir+reference_files[ind])
    #print(reference_image)
    reference_image = reference_image.astype(np.float32)
    reference_image = cv.normalize(reference_image, None, 0, 255, cv.NORM_MINMAX)

    rmargin = round(margin/scale)
    new_h = reference_image.shape[0]+round(2*rmargin)
    new_w = reference_image.shape[1]+round(2*rmargin)
    padded_reference = np.full((new_h, new_w), 255, dtype=np.uint8)
    padded_reference[rmargin:rmargin + reference_image.shape[0], rmargin:rmargin + reference_image.shape[1]] = reference_image
    reference_image = padded_reference
    # reference_image= cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)
    # reference_image = cv.normalize(reference_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    #2. transform with R, T and S
    new_h = transform_image.shape[0]+2*margin
    new_w = transform_image.shape[1]+2*margin
    padded_transform = np.full((new_h, new_w), 255, dtype=np.uint8)
    padded_transform[margin:margin + transform_image.shape[0], margin:margin + transform_image.shape[1]] = transform_image
    transform_image = padded_transform
    transform_image_ori = transform_image

    #print(R.shape, T.shape, type(R))
    transform_image = transform_img(transform_image_ori, R, T, scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    
    im = ax1.imshow(transform_image, cmap='gray')
    ax1.set_title('transform Image')

    reference_image_ori = reference_image
    reference_image = MLX_process(reference_image)
    ax2.imshow(reference_image)
    ax2.set_title('reference Image')


    # Create axes for the sliders=====================
    ax_angle = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_xshift = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_yshift = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_scale = plt.axes([0.25, 0.10, 0.65, 0.03])

    # Create sliders
    angle_slider = Slider(ax_angle, 'Angle', -math.pi/2, math.pi/2, valinit=math.asin(R[1][0]))
    xshift_slider = Slider(ax_xshift, 'X Shift', T[0]-20, T[0]+20, valinit=T[0])
    yshift_slider = Slider(ax_yshift, 'Y Shift', T[1]-20, T[1]+20, valinit=T[1])
    scale_slider = Slider(ax_scale, 'Scale', scale-0.2, scale+0.2, valinit=scale)


    # Attach the update function to the sliders
    angle_slider.on_changed(update)
    xshift_slider.on_changed(update)
    yshift_slider.on_changed(update)
    scale_slider.on_changed(update)

    # sliders done==============================================


    #curser===================================================
    #1. add cursor to show corresponding points
    cursor1, = ax1.plot([], [], 'r+')
    cursor2, = ax2.plot([], [], 'r+')

    cursor11, = ax1.plot([], [], 'r+')
    cursor21, = ax2.plot([], [], 'r+')
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    #2. layout and show
    plt.tight_layout()
    plt.title("Transformed reference Image")
    plt.show()
    # cursor done=================================================

    #plt.clear()
    # show overlapping image
    transform_image = transform_img(transform_image_ori, R, T, scale)
    plt.imshow(transform_image, cmap='gray', alpha=0.5)
    plt.imshow(reference_image_ori, cmap='gray', alpha=0.5)
    plt.xlim(-10, 250)
    plt.ylim(200, -10)
    plt.show()

    to_save = input("save R, T and s? y/n")
    if to_save=="y":
        dump_yaml(R, T, scale, RTSfileDst)

    
