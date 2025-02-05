import matplotlib
matplotlib.use('TkAgg')  # Force a different backend
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot([1, 2, 3], [4, 5, 6])
ax2.plot([1, 2, 3], [6, 5, 4])
plt.show()