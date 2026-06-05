import matplotlib.pyplot as plt
import matplotlib as mpl
import os

OUT_PATH = "/path/to/SelfRDB_128/hot_colorbar_vertical_0to1.png"

HEAT_CMAP = "hot"
HEAT_VMIN = 0.0
HEAT_VMAX = 1.0
DPI = 600

fig, ax = plt.subplots(figsize=(0.8, 5))

norm = mpl.colors.Normalize(vmin=HEAT_VMIN, vmax=HEAT_VMAX)
cbar = mpl.colorbar.ColorbarBase(
    ax,
    cmap=plt.get_cmap(HEAT_CMAP),
    norm=norm,
    orientation='vertical'
)

cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Saved vertical colorbar to:", OUT_PATH)