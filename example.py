import matplotlib.pyplot as plt
import numpy as np

def on_mouse_move(event):
    if event.inaxes == ax:
        print(type(cursor))
        cursor.set_data(event.xdata, event.ydata)
        plt.draw()

image = np.random.rand(100, 100)
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
cursor, = ax.plot([], [], 'r+')  # Red cross cursor

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
plt.show()