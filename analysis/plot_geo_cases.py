import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import SVG, display

areas = ['17097', '32003']

for area in areas:

    print(f"Processing area: {area}")

    shapefile_path = f"/Users/hideki-h/Desktop/実験用データ/ComOD-dataset/assets/Boundaries_Regions_within_Areas/{area}/{area}.shp"

    gdf = gpd.read_file(shapefile_path)
    print(gdf.head())

    svg_path = f"figs/{area}_map.svg"

    fig, ax = plt.subplots()
    gdf.plot(ax=ax, edgecolor="black", facecolor="none")

    ax.set_axis_off() # 軸・枠を消す
    fig.patch.set_alpha(0) # 背景透明

    plt.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    display(SVG(filename=svg_path))