import matplotlib.pyplot as plt

from sklearn import datasets, manifold

sr_points, sr_color = datasets.make_swiss_roll(n_samples=1500, random_state=0)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)
plt.savefig('swissroll.png')



sr_isomap = manifold.Isomap(n_components=2).fit_transform(
    sr_points
)

plt.scatter(sr_isomap[:, 0], sr_isomap[:, 1], c=sr_color)
plt.title('Isomap embedding of Swiss Roll')
plt.savefig('isomap_swissroll.png')