from eomaps import Maps
import numpy as np
import mapclassify
from itertools import pairwise
from pathlib import Path
import pandas as pd
from matplotlib.colors import ListedColormap
import textwrap
from functools import lru_cache

Maps.config(log_level="debug")

@lru_cache()
def getdata(f, block_size=(10, 10)):
    data = Maps.read_file.GeoTIFF(f)
    # if f.stem.startswith("SIG"):
    #     data["encoding"]["scale_factor"] = 0.01

    # --------------- Resampling (Interpolated)
    # data["data"] = zoom(data["data"].filled(), 0.1, order=0)
    # data["x"] = zoom(data["x"], 0.1)
    # data["y"] = zoom(data["y"], 0.1)

    # --------------- Resampling (mean/median)
    d = data["data"]
    m, n = d.shape

    if f.stem.startswith("SIG"):
        d = np.ma.mean(d.reshape(m//block_size[0], block_size[0], n//block_size[1], block_size[1]), axis=(1,3))
    else:
        d = np.ma.mean(d.reshape(m//block_size[0], block_size[0], n//block_size[1], block_size[1]), axis=(1,3))

    data["data"] = d#.filled(0)
    data["x"] = np.nanmean(data["x"].reshape(len(data["x"])//block_size[0], block_size[0]), axis=1)
    data["y"] = np.nanmean(data["y"].reshape(len(data["y"])//block_size[1], block_size[1]), axis=1)



    d = data["data"]/10
    x, y = np.meshgrid(data["x"], data["y"])

    mask = ~d.mask
    d, x, y = d[mask], x.T[mask], y.T[mask]
    crs = Maps.CRS.Equi7_AS


    return d, x, y, crs






# %%

level = "expert"

shape = "ellipses"
cb_label = "Rückstreu Koeffizient $\sigma^0 [dB]$"


cmap=ListedColormap(["b", "c", "orange", "r"])


if level == "expert":
    agg = (200, 200)
    number_fontsize = 5
    number_fontweight = "normal"
    #extent = (7198413.619392307, 7964204.529186169, 2605835.9214183614, 3569813.5134464856)
    extent = (7045022.633840765, 8163030.824519138, 2535898.0540122753, 3943246.7568167606)

else:
    agg = (500, 500)
    number_fontsize = 10
    number_fontweight = "bold"
    #extent = (7168325.18788458, 7990240.840329152, 2549412.708590865, 3584040.1262851)
    extent = (7045022.633840765, 8163030.824519138, 2535898.0540122753, 3943246.7568167606)



bins = [-15, -12.5, -10, -7.5]
classify=dict(scheme="UserDefined", bins=bins)

vmin=-17.5
vmax=-7.5


def create_ann_layer(m_parent, layer_name, d, x, y, crs, add_colorbar=False):
    m = m_parent.new_layer(layer_name)

    cdata = mapclassify.classify(d.ravel(), **classify)

    m.set_data(d, x, y, crs=crs)
    m.set_classify.UserDefined(bins=cdata.bins)
    m.set_classify_specs(**classify)

    getattr(m.set_shape, shape)(radius=2e4, radius_crs=Maps.CRS.Equi7_AS)

    m.set_shape.rectangles()


    m.plot_map(cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.25
               )

    # if add_colorbar:
    #     cb = m.add_colorbar(hist_bins="bins", extend="neither")
    #     cb.set_labels(cb_label=cb_label, fontsize=14)
    #     cb.set_labels(hist_label="# points", fontsize=10)

    # -----------------
    m_ann = m.new_layer(f"{layer_name}_ann",
                        inherit_data=True,
                        inherit_shape=True,
                        inherit_classification=True
                        )
    m_ann.set_classify_specs(**classify)
    m_ann.plot_map(fc=(1,1,1,.5), ec="k", lw=0.5)

    nbins = (len(bins) - 1)  # values larger than the upper bin = nbin as value
    m_ann.add_annotation(
        text=lambda ID, **kwargs: f"{cdata.yb[ID] if cdata.yb[ID] <= nbins else nbins}",
        ID=range(cdata.yb.size),
        ha="center", va="center", xytext=(0, 0),
        bbox=dict(fc="none", ec="none"),
        arrowprops=None,
        layer=m_ann.layer,
        permanent="fixed",
        fontweight=number_fontweight,
        fontsize=number_fontsize,
        )

    if add_colorbar:
        cb = m_ann.add_colorbar(hist_bins="bins", extend="neither")
        cb.set_labels(cb_label=cb_label, fontsize=14)
        cb.set_labels(hist_label="", fontsize=10)
        cb.ax_cb_plot.tick_params(left=False, labelleft=False)
        cb.ax_cb_plot.grid(axis="y")


        # add number indicators to classes
        cby = np.sum(cb.ax_cb.get_ylim()) * 0.5
        cbx = [vmin, *bins]
        for i, cbx in enumerate(pairwise(cbx)):
            cb.ax_cb.text(np.mean(cbx), cby, i,
                               fontsize=20, fontweight="bold",
                               ha="center", va="center",
                               bbox=dict(
                                   boxstyle='circle,pad=0.5',
                                   fc='w', lw=2, ec="k",
                                   ),
                               )

def create_full_layer(m, layer_name, f):
    m_sig = m.new_layer(layer_name)
    full_data = Maps.read_file.GeoTIFF(f)
    full_data["data"] = full_data["data"] / 10
    m_sig.set_data(**full_data)
    m_sig.set_classify_specs(**classify)
    m_sig.set_shape.shade_raster()
    m_sig.plot_map(cmap=cmap, vmin=vmin, vmax=vmax)
    #m_sig.add_colorbar()


f = Path(r"D:\ERN\ERN23\resampled_100m\SIG0_20220514_resampled.tif")
f1 = Path(r"D:\ERN\ERN23\resampled_100m\SIG0_20220830_resampled.tif")

m = Maps(ax=121, crs=3857, figsize=(12, 10))
m2 = m.new_map(ax=122, crs=3857)
m.join_limits(m2)


gdf = m.add_feature.physical.rivers_lake_centerlines.get_gdf()



for mi in (m, m2):
    mi.all.add_feature.preset.coastline(lw=0.25)
    mi.all.add_feature.preset.ocean(alpha=0.35)

    mi.add_wms.ESRI_ArcGIS.SERVICES.NatGeo_World_Map.add_layer.xyz_layer(layer="wms")

    mi.add_gdf(gdf, lw=3, ec="b", fc="none", layer="overlay")

bbox = dict(boxstyle='round', facecolor='.8', ec="k", lw=2)
for text, usem in zip(("14 Mai 2022", "30 August 2022"), (m, m2)):
    usem.text(0.5, 0, text, transform=usem.ax.transAxes,
              layer="all", fontsize=14, fontweight="bold",
              bbox=bbox)

m.text(0.5, .96, "Findest du die Flut?\nFüll die Felder mit den Farben und vergleiche die Bilder!", fontweight="bold",
       fontsize=18, transform=m.f.transFigure, layer="all")

create_ann_layer(m, "flood", *getdata(f, agg), add_colorbar=True)
#create_full_layer(m, f.stem.split("_")[1] + "_full", f)

create_ann_layer(m2, "flood", *getdata(f1, agg))
#create_full_layer(m2, f1.stem.split("_")[1] + "_full", f1)


gl = m.all.add_gridlines(([65, 67.5, 70, 72.5], [20,25,30,35]), layer="grid", c=".5")
gl.add_labels(where="bl", exclude=[67.5, 70], offset=15, c=".5")
gl = m2.all.add_gridlines(([65, 67.5, 70, 72.5], [20,25,30,35]), layer="grid", c=".5")
gl.add_labels(where="br", exclude=[67.5, 70], offset=15, c=".5")

#m.fetch_layers()

m.show_layer("wms", "grid", "flood_ann", ("overlay", .35))

m.add_logo()
m.add_logo(r"C:\Users\rquast\Desktop\_delete\geo_blue_white.png")

layout = {
    "figsize": [13.29, 9.27],
    "0_map": [0.05274, 0.18638, 0.40636, 0.73335],
    "1_map": [0.53961, 0.18568, 0.40713, 0.73475],
    "2_cb": [0.42, 0.0215, 0.575, 0.11469],
    "2_cb_histogram_size": 0,
    "3_logo": [0.955, 0.00717, 0.04, 0.02366],
    "4_logo": [0.8625, 0.93188, 0.07863, 0.05376],
}
m.apply_layout(layout)
m.set_extent(extent, 3857)

# %

# m.BM.remove_artist(t)
# m.BM.remove_artist(t2)

t = m.text(0.065, 0.09,
           textwrap.indent(
           textwrap.fill("Mit Hilfe des Sentinel-1 Satelliten wird das Rückstreuverhalten von Mikrowellen auf dem gesamten Planeten gemessen.",
                         65)
           + "\n\n" +
           textwrap.fill("Die zeitliche Veränderung des Signals kann dazu verwendet werden geflutete Flächen zu identifizieren!",
           60)
           , "     ")
           ,
           ha="left", va="center",
           layer="all",
            bbox=dict(
                boxstyle='round,pad=0.5',
                fc=(1, .5, 0, .25), lw=2, ec="k",
                ),

           )

t2 = m.text(0.065, 0.09, "?",
           ha="right", va="center",
           layer="all",
            bbox=dict(
                boxstyle='circle,pad=0.3',
                fc=(1, .5, 0), lw=2, ec="k",
                ),
            fontsize=20,
            fontweight="bold"
           )

import matplotlib.pyplot as plt
plt.pause(0.05)

#m.savefig(rf"D:\ERN\flood_v1_{level}.png", dpi=200)
