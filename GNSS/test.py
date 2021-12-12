import numpy as np
from matplotlib import pyplot as plt

# throw three darts:
rad = np.asarray([10000, 11000, 9400]) # consider these as distances from bull's eye in 10*µm
azi = np.asarray([352, 0, 10]) # in degrees
azi = np.deg2rad(azi) # conversion into radians

fig = plt.gcf()

axes_coords = [0, 0, 1, 1] # plotting full width and height

ax_polar = fig.add_axes(axes_coords, projection='polar', label="ax polar")
# ax_polar = fig.add_axes(axes_coords, projection='polar')
ax_polar.patch.set_alpha(0)
ax_polar.scatter(azi, rad)
ax_polar.set_ylim(0, 17000)
ax_polar.set_xlim(0, 2*np.pi)
ax_polar.set_theta_offset(0.5*np.pi) # 0° should be on top, not right
ax_polar.set_theta_direction(direction=-1) # clockwise



pic = plt.imread("bj.png")
ax_image = fig.add_axes(axes_coords, label="ax image")
# ax_image = fig.add_axes(axes_coords)
ax_image.imshow(pic, alpha=.5)
ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image

plt.show()