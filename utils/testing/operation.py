from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load image
image = Image.open("test0.jpg")  
plt.imshow(image)
plt.show()

# Create a draw object
draw = ImageDraw.Draw(image)

# Draw the dot (ellipse with radius=2)
def draw_ellipse(dot_position, dot_radius1, dot_radius2):
    bbox = [
        dot_position[0] - dot_radius1, 
        dot_position[1] - dot_radius2,
        dot_position[0] + dot_radius1, 
        dot_position[1] + dot_radius2
    ]
    draw.ellipse(bbox, fill=dot_color)

# Add dot at (x=114, y=514) with RGB color
dot_position = (1315, 1202)  # (x, y)
dot_color = (250, 250, 80)  # RGB
dot_radius1 = 30
dot_radius2 = 30
draw_ellipse(dot_position, dot_radius1, dot_radius2)

# Add dot at (x=114, y=514) with RGB color
dot_position = (500, 1000)  # (x, y)
dot_color = (250, 250, 80)  # RGB
dot_radius1 = 30
dot_radius2 = 60
draw_ellipse(dot_position, dot_radius1, dot_radius2)

#rectangle
rec_color = (250, 250, 80)
top_left = (45, 1200)
width, height = 100, 300  # Adjust size as needed
bottom_right = (top_left[0] + width, top_left[1] + height)
draw.rectangle(
    [top_left, bottom_right],
    fill=rec_color,
    outline=None  # Optional: add border with outline=(R,G,B)
)

# Save the result
image = np.array(image)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("test1_dotdot_rec.jpg", image)