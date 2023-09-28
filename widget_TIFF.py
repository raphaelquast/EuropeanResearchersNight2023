"""
EOmaps widget for the European Researchers Night 2023

contact: raphael.quast@geo.tuwien.ac.at
"""

from pathlib import Path
from functools import lru_cache
from argparse import ArgumentParser
import warnings

from eomaps import Maps
from eomaps.cb_container import CallbackContainer

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.axes as axes

import pandas as pd

# %% general settings and definitions

# set global Maps config
Maps.config(use_interactive_mode=False, log_level="debug")
# ignore warnings (to show only eomaps messages in the log)
warnings.filterwarnings("ignore")

# %%% parse command line args
parser = ArgumentParser()
parser.add_argument("-f", "--fetch_layers", default=False,
                    help="fetch all layers immediately")
commandline_args = parser.parse_args()

# %%% matplotlib style funcs

class StaticColorAxisBBox(mpatches.FancyBboxPatch):
    def set_edgecolor(self, color):
        if hasattr(self, "_original_edgecolor"):
            return
        self._original_edgecolor = color
        self._set_edgecolor(color)

    def set_linewidth(self, w):
        super().set_linewidth(1.5)

class FancyAxes(axes.Axes):
    name = "fancy_box_axes"
    _edgecolor: str

    def __init__(self, *args, **kwargs):
        self._edgecolor = kwargs.pop("edgecolor", None)
        super().__init__(*args, **kwargs)

    def _gen_axes_patch(self):
        return StaticColorAxisBBox(
            (0, 0),
            1.0,
            1.0,
            boxstyle="round, rounding_size=0.025, pad=0",
            edgecolor=self._edgecolor,
            linewidth=5,
            mutation_aspect=self.bbox.width / self.bbox.height
        )

def set_mpl_defaults():
    # manually configure keyboard shortcuts
    CallbackContainer._remove_default_keymaps = lambda *args, **kwargs: None
    # remove all default keymaps
    for key, val in mpl.rcParams.items():
        if key.startswith("keymap"):
            mpl.rcParams[key] = []
    # add custom keymaps
    mpl.rcParams["keymap.fullscreen"] = ["f"]
    mpl.rcParams["keymap.home"] = ["h"]
    mpl.rcParams["keymap.quit"] = ["ctrl+w", "cmd+w"]
    mpl.rcParams["keymap.zoom"] = ["o"]
    mpl.rcParams["keymap.pan"] = ["p"]

    # set default text color to white
    mpl.rcParams["text.color"] = "w"

def style_toolbar(tb):
    tb.setStyleSheet("""
                     QToolBar {
                         background-color: black;
                         border: 0px;
                         }
                     QToolBar:separator {
                         border: 0px;
                         }
                     QToolBar:handle {
                         border: 0px;
                         }

                     QToolButton {
                         background-color: rgb(50, 50, 50);
                         border: 0px;
                         border-radius: 10px;
                         padding: 2px;
                         margin: 2px;
                         
                         }
                      QToolButton:hover {
                          background-color: rgb(150, 150, 150);
                          }
                      QToolButton:checked {
                          background-color: rgb(250, 100, 100);
                          }
                      """)
        
    tb.setVisible(True)

    tb._actions["save_figure"].setVisible(False)
    tb._actions["edit_parameters"].setVisible(False)
    tb._actions["configure_subplots"].setVisible(False)
    tb._actions["back"].setVisible(False)
    tb._actions["forward"].setVisible(False)


# set global matplotlib configs
set_mpl_defaults()


#%% load datasets

# folder-structure:
#     low-res images:  ERN23/resampled_100m/SIG0_20220514_resampled.tif
#     low-res flood:   ERN23/resampled_100m/GFM-FLOOD_20220514_resampled.tif
#     high-res images: ERN23/orig/SIG0_20220514.tif
#     high-res flood:  ERN23/orig/GFM-FLOOD_20220514.tif

@lru_cache()
def getdata(f):
    # load data for plots
    data = Maps.read_file.GeoTIFF(f)
    if f.stem.startswith("SIG"):
        data["encoding"]["scale_factor"] = 0.1

    #--------------- Resampling (Interpolated)
    # from scipy.ndimage import zoom
    # scale = 0.01
    # data["data"] = zoom(data["data"].filled(), scale, order=0)
    # data["x"] = zoom(data["x"], scale)
    # data["y"] = zoom(data["y"], scale)
    
    return data

try:
    __file__
except NameError:
    # just for manual debugging... use hard-coded paths from various pcs
    for f in ("D:/ERN",
              "C:/Users/Admin/Projects/EuropeanResearchersNight2023",            
              ):
        __file__ = Path(f)
        if not __file__.exists():
            continue
    else:
        raise AssertionError("Oh no... none of the hardcoded paths exists...")
        

folder = Path(__file__).parent / "ERN23"/ "resampled_100m"
ts_path = Path(__file__).parent / "ERN23"/ "timeseries"
feature_path = Path(__file__).parent / "ERN23"/ "features"

# identify all GeoTIFF files
files = [i for i in folder.iterdir() if i.suffix == ".tif"]

# identify layer names based on available files
layernames = []
for i in files:
    if i.stem.startswith("SIG"):
        date = i.stem[5:].replace('_resampled', '')

        layernames.append(f"{date[:4]} {date[4:6]} {date[6:]}")
    else:
        layernames.append(f"_{i.stem}")

# load data for timeseries plots
ts_data = dict(x=[], y=[], data=[], f=[])
for f in ts_path.iterdir():
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    ts_data["x"].append(df.iloc[0].x)
    ts_data["y"].append(df.iloc[0].y)
    ts_data["data"].append(int(f.stem.replace("ts", "")))
    ts_data["f"].append(f)

ts_data = pd.DataFrame(ts_data)

# %% callbacks

# define colormaps
cmap_flood = ListedColormap(["none", "r"])
cmap_flood.set_under((0,0,0,0))
cmap_flood.set_over((0,0,0,0))

cmap_sig = plt.get_cmap("viridis")

# callback to lazily load data-layers
def load_and_plot_data(m, f):
    m.set_data(**getdata(f))

    if f.stem.startswith("SIG"):
        m.set_shape.shade_raster(aggregator="mean")
        m.plot_map(vmin=-20, vmax=5, cmap=cmap_sig, set_extent=False)
    else:
        m.set_shape.shade_raster(aggregator="first")
        m.plot_map(vmin=0, vmax=1, cmap=cmap_flood, set_extent=False)

    # peek on the flood-classification layer on right click
    m.cb.click.attach.peek_layer(
        "_".join(("_GFM-FLOOD", *f.stem.split("_")[1:])), how=0.25, shape="round",
        button=3,
        )

    # annotate sig0 value on left click button
    m.cb.pick.attach.annotate(
        bbox=dict(fc="k"),
        text = lambda val, **kwargs: f"$\sigma^0[dB]=${val:.2f}",
        button=2
        )
    # add a marker to identify the clicked pixel on left click button
    m.cb.pick.attach.mark(
        ec="r", fc="none",
        button=2
        )

# callbacks to plot timeseries
artists = dict()
def plot_timeseries(ID, ax, **kwargs):
    ax.set_visible(True)
    if ID not in artists:
        f = ts_data.loc[ID].f
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        l, = ax.plot(df.backscatter, c=f"C{ID}")
        artists[ID] = l

    for key, val in artists.items():
        if key == ID:
            val.set_visible(True)
        else:
            val.set_visible(False)
    ax.autoscale()
    ax.autoscale_view()

# callback to add lines to indicate dates of the images in the timeseries
indicators = dict()
def add_time_indicator(m, **kwargs):
    layer = m.BM.bg_layer
    if layer not in indicators:
        try:
            d = pd.to_datetime(layer, format="%Y %m %d")
            indicators[layer] = ax.plot([d, d], [-35, -5], ls=(0, (5,5)), lw=1, c="w", zorder=-1)[0]
        except Exception:
            pass

    for l, a in indicators.items():
        if l == layer:
            a.set_visible(True)
        else:
            a.set_visible(False)


# callback to show/hide timeseries on hover
def show_hide_timeseries(pos, m, ax, show_timeseries=False, **kwargs):
    found = m.tree.query(pos)
    if found is not None:
        marker = m.add_marker(
            xy=m._data_manager._get_xy_from_ID(found), xy_crs=m.data_specs.crs,
            permanent=False, fc="none", ec="r", alpha=0.5, zorder=9999,
            shape="scatter_points", radius=200
            )
        m.cb.move.add_temporary_artist(marker)

        if show_timeseries:
            plot_timeseries(found, ax=ax)
    else:
        if show_timeseries:
            ax.set_visible(False)

# callback to show/hide annotations on hover
annotations = dict()
def add_annotations(pos, m, **kwargs):
    found = m.tree.query(pos)
    if found is not None:
        xy = m._data_manager._get_xy_from_ID(found)

        marker = m.add_marker(
            xy=xy, xy_crs=m.data_specs.crs,
            permanent=False, fc="none", ec="r", alpha=0.5, zorder=9999,
            shape="scatter_points", radius=200
            )
        m.cb.move.add_temporary_artist(marker)
        
        if found not in annotations:
            annotation = m.add_annotation(
                xy=xy, xy_crs=m.data_specs.crs,
                text=m.data.loc[found].text,
                color="k",
                arrowprops=dict(arrowstyle="fancy", facecolor="w"),
                bbox=dict(boxstyle="round", fc="w",lw=2),
                xytext=(0.5, 0.05),
                ha="center", va="center",
                textcoords="figure fraction",
                fontsize=15,
                permanent=False,
                )
            annotations[found] = annotation
    # make sure only a single annotation is shown at a time
    # (and no flickering occurs if the same point is hovered multiple times)
    for key in list(annotations):
        if key != found:
            a = annotations.pop(key)
            try:
                m.BM.remove_artist(a)
                a.remove()
            except Exception:
                pass

# %% create the map
m = Maps(crs=Maps.CRS.Equi7_AS, facecolor="k")
# hide the spines (e.g. boundary lines)
m.ax.spines["geo"].set_visible(False)
# style the toolbar
style_toolbar(m.f.canvas.toolbar)

# set the extent of the map
m.set_extent(
    (1037084.5566022274, 2433790.9845247325, 2502130.3252743697, 3336077.3654134865),
    Maps.CRS.Equi7_AS)

# add a title
m.all.text(x=0.5, y=1,
           s="Überschwemmungen aus dem Weltraum beobachten mit Sentinel-1!",
           color="w",
           fontweight="bold",
           ha="center", 
           va="top",
           fontsize=20,
           transform=m.f.transFigure
           )

# %%% add basic map features 
# (cache reprojected versions for faster access)
if not feature_path.exists():
    feature_path.mkdir()

cached_features = [i.stem for i in feature_path.iterdir()]

feature_kwargs = dict(
    physical_coastline=dict(ec="w", lw=2, fc="none"),
    physical_ocean=dict(hatch="/////", lw=0.5, alpha=0.5, ec=".6", fc="none"),
    cultural_admin_0_countries=dict(ls="--", ec="w", lw=1, fc="none"),
    )

for key, args in feature_kwargs.items():
    category, name = key.split("_", 1)
    if name in cached_features:
        m.all.add_gdf(feature_path / f"{name}.gpkg", **args)
    else:
        print(f"... caching feature {category} - {name}")
        # fetch data, reproject to plot crs and save for later use
        feature = getattr(getattr(m.add_feature, category), name)
        
        gdf = m._handle_gdf(
            feature.get_gdf(scale=50, what="geoms_intersecting"), 
            reproject="cartopy")
        gdf.to_file(feature_path / f"{name}.gpkg")
        m.all.add_gdf(gdf, **args)

# %%% add all (lazy) data layers
i = 0
for f, name in zip(files, layernames):
    # low-res layers
    m.on_layer_activation(
        load_and_plot_data,
        layer=name,
        f=f
        )

    # high-res layers
    m.on_layer_activation(
        load_and_plot_data,
        layer=name + "_HR",
        f=f.parent.parent / "orig" / (f.stem.replace("_resampled", "") + f.suffix)
        )

    # add keypress callbacks to switch between the layers
    if not name.startswith("_"):
        i += 1
        m.all.cb.keypress.attach.switch_layer(key=f"{i}", layer=name)
        m.all.cb.keypress.attach.switch_layer(key=f"ctrl+{i}", layer=name + "_HR")

# %%% add time-indicator callbacks for timeseries
# (e.g. show/hide indicator on each layer change)
for l in layernames:
    if l.startswith("20"):
        m.on_layer_activation(add_time_indicator, layer=l, persistent=True)

# %%% add a selector widget to indicate the active layer
sel = m.util.layer_selector(
    layers = layernames,
    draggable=False,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.9),
    facecolor=(0, 0, 0, 0.8),
    fontsize="large",
    markerscale=2,
    title="Beobachtungsdatum",
    title_fontsize="x-large",
    labelspacing = 1.5,
    )
sel.leg.get_frame().set_facecolor((0, 0, 0, .5))

# %%% add info-text on the active controls of the map
x0, y0 = 0.05, 0.2

m.all.text(
    x0, y0,
    "Steuerung:",
    color="w", ha="left", va="top",
    fontsize=15, fontweight="bold"
    )

m.all.text(
    x0, y0 - 0.035,
    "\n".join((
    "● 1, 2, 3, 4, 5: Zeige daten für die ausgewählten Tage",
    "    (ctrl + # für hochaufgelößtes bild)",
    "",
    "● Rechts-klick: Zeige die Flut-klassifizierung",
    "",
    "● P: Aktiviere das Pan/Zoom tool",
    "● O: Aktiviere das Rechteck Zoom tool"

    ))
    ,
    color="w", ha="left", va="top",
    fontsize=12,
    )


# %% add annotations for interesting points on the map
points_cmap = ListedColormap(["r", "b"])

points = [
    dict(x=68.2073, y=27.5588, val=0, text="Stadt: Larkana\n(~500 000 Einwohner)"),
    dict(x=67.6392, y=26.4226, val=1, text="Manchar See\nParkistans größter Süßwassersee!"),
    dict(x=68.3588, y=25.3912, val=0, text="Stadt: Hyderabad\n(~9.5 millionen Einwohner",),
    dict(x=67.0621, y=24.9006, val=0, text="Stadt: Karachi\nEine der größten Städte der Welt!\n(~20 millionen Einwohner)")
    ]

points = pd.DataFrame.from_records(points)

# create a new map for the points
m_cities = m.new_layer("all")
m_cities.set_data(points, "x", "y", crs=4326, parameter="val")
m_cities.set_shape.scatter_points(size=100)
m_cities.plot_map(zorder=9999, cmap=points_cmap, ec="none", layer="_dummy")

m_cities.cb.move.attach(add_annotations, m=m_cities)
# initialize the picker to make points pickable without a pick callback
# (e.g. to init the search-tree)
m_cities.cb.pick.set_props(search_radius=5000)
m_cities.cb.pick._init_picker()

# plot empty circles to indicate the locations
mc2 = m_cities.new_layer(inherit_data=True, inherit_classification=True, inherit_shape=True)
mc2.plot_map(fc="none", ec="k", zorder=99, lw=0.5)


# %% add timeseries plots

# create axes for the timeseries
ax = m.f.add_subplot(zorder=99999,
                     facecolor=(0,0,0,.6),
                     edgecolor="w",
                     axes_class=FancyAxes)
ax.set_visible(False)
for key, val in ax.spines.items():
    val.set_visible(False)

ax.tick_params(axis="x", direction="in", pad=-15, color="w", labelcolor="w")
ax.tick_params(axis="y", direction="in", pad=-30, color="w", labelcolor="w")
ax.set_ylabel("$\sigma^0$[dB]", color="w", fontsize=12)
ax.set_title("Sentinel-1 backscatter timeseries", color="w", fontsize=12, fontweight="bold")

ax.grid(lw=0.25, color="w", ls=":")
ax.set_ylim(-32.5, -2.5)
# disable navigation on the timeseries-axes (avoid overlapping axes zoom bug)
ax.set_navigate(False)


# %%% plot points to show/hide timeseries on hover
m_ts = m.new_layer("all")
m_ts.set_data(ts_data, "x", "y", parameter="data", crs=Maps.CRS.Equi7_AS)
m_ts.set_shape.scatter_points(size=100)
m_ts.plot_map(zorder=999,
              ec="k",
              cmap=ListedColormap([f"C{i}" for i in range(len(ts_data))]))
# initialize the picker to make points pickable without a pick callback
# (e.g. to init the search-tree)
m_ts.cb.pick.set_props(search_radius=5000)
m_ts.cb.pick._init_picker()
m_ts.cb.click.attach(show_hide_timeseries, show_timeseries=True, m=m_ts, ax=ax)
m_ts.cb.move.attach(show_hide_timeseries, show_timeseries=True, m=m_ts, ax=ax)


# %% add logos
m.add_logo(Path(__file__).parent / "geo_blue_white.png")
m.add_logo()

# %% add a scalebar
scb = m.all.add_scalebar(
    auto_position=(.05, .3),
    autoscale_fraction=.08,
    size_factor=0.35,
    rotation=-18.0
    )

# %% set layout
layout = {
    "figsize": [15.36, 9.23],
    "0_map": [0.025, 0.01952, 0.95, 0.94394],
    "1_": [0.5375, 0.09339, 0.4375, 0.23346],
    "2_logo": [0.93871, 0.94648, 0.055, 0.04365],
    "3_logo": [0.9425, 0.01248, 0.05, 0.03433],
}

m.apply_layout(layout)

# %% fetch layers, toggle fullscreen and show
if bool(commandline_args.fetch_layers) is True:
    m.fetch_layers([i for i in m._get_layers() if "_HR" not in i])

m.f.canvas.manager.full_screen_toggle() # toggle fullscreen mode

m.show_layer(layernames[len(layernames)//2])

plt.show()





