import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import rotate, shift, zoom

# Load or create an example image
image = np.random.rand(100, 100)  # Replace this with your image

# Create the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)  # Adjust layout to make room for sliders

# Display the initial image
im = ax.imshow(image, cmap='gray')

# Create axes for the sliders
ax_angle = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_xshift = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_yshift = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_scale = plt.axes([0.25, 0.10, 0.65, 0.03])

# Create sliders
angle_slider = Slider(ax_angle, 'Angle', -30, 30, valinit=0)
xshift_slider = Slider(ax_xshift, 'X Shift', -100, 100, valinit=0)
yshift_slider = Slider(ax_yshift, 'Y Shift', -100, 100, valinit=0)
scale_slider = Slider(ax_scale, 'Scale', 0.1, 2.0, valinit=1.0)

# Function to update the image based on slider values
def update(val):
    # Get current slider values
    angle = angle_slider.val
    xshift = xshift_slider.val
    yshift = yshift_slider.val
    scale = scale_slider.val

    # Apply transformations
    transformed_image = rotate(image, angle, reshape=False)  # Rotate
    transformed_image = shift(transformed_image, (yshift, xshift))  # Shift
    transformed_image = zoom(transformed_image, scale)  # Scale

    # Update the displayed image
    im.set_data(transformed_image)
    fig.canvas.draw_idle()

# Attach the update function to the sliders
angle_slider.on_changed(update)
xshift_slider.on_changed(update)
yshift_slider.on_changed(update)
scale_slider.on_changed(update)

# Show the plot
plt.show()