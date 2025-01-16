import matplotlib.pyplot as plt
import numpy as np

# Assuming 'thermal_image' is your grayscale thermal image
thermal_image = np.random.rand(100, 100)  # Replace with your image

plt.imshow(thermal_image, cmap='hot')
cbar = plt.colorbar()
cbar.set_label('Temperature')
plt.axis('off')
plt.show()