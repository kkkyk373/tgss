import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import SVG, display
import os
areas = ['17097', '32003']

for area in areas:

    print(f"Processing area: {area}")

    base_dir = os.environ.get("SHAPEFILE_BASE_DIR", "assets/Boundaries_Regions_within_Areas")
    shapefile_path = os.path.join(base_dir, area, f"{area}.shp")

    gdf = gpd.read_file(shapefile_path)
    print(gdf.head())

    svg_path = f"figs/{area}_map.svg"

    fig, ax = plt.subplots()
    gdf.plot(ax=ax, edgecolor="black", facecolor="none")

    ax.set_axis_off() # 軸・枠を消す
    fig.patch.set_alpha(0) # Make background transparent

    plt.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    display(SVG(filename=svg_path))