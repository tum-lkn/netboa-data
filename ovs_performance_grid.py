#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


#======================================================================================
# Specify your preambles. Should be made more generic in the future. Add subcategories.
# Names used to identify your setup.
#======================================================================================
DISS = 'DISS'
IEEE = 'IEEE'
ACM = 'ACM'

TEX_IEEE_PREAMBLE = [
    r'\usepackage{mathptmx}',
    r'\usepackage[T1]{fontenc}',
    # r'\usepackage{bm}',
    r'\usepackage{amsmath}',
    # r'\usepackage{upgreek}',
    # r'\usepackage{textcomp}'
    r'\usepackage{siunitx}',
    r'\sisetup{range-units=single,'
    # r'        range-phrase = \,--\,,'
    # r'        list-units=single,'
    # r'        number-unit-product = \,,'
    # r'        separate-uncertainty = true,'
    # r'        multi-part-units = single,'
    # r'        detect-weight,'
    # r'        detect-display-math,'
    # r'        detect-inline-weight=math,'
    r'        per-mode=symbol,'
    r'        }',
    r'\DeclareSIUnit{\msg}{msg}',
    r'\DeclareSIUnit{\second}{s}'
]

PGF_IEEE_PREAMBLE = [
    r'\usepackage{amsmath}',
    #r'\usepackage{bm}',
    #r'\renewcommand{\v}{\bm}',
    r'\usepackage{unicode-math}',  # unicode math setup
    #r'\setmathfont{XITS Math}',
    #r'\setmainfont{texgyretermes}',
    r'\usepackage{siunitx}',
    r'\sisetup{range-units=single,'
    #r'        range-phrase = \,--\,,'
    #r'        list-units=single,'
    #r'        number-unit-product = \,,'
    #r'        separate-uncertainty = true,'
    #r'        multi-part-units = single,'
    #r'        detect-weight,'
    #r'        detect-display-math,'
    #r'        detect-inline-weight=math,'
    r'        per-mode=symbol,'
    r'        }',
    r'\DeclareSIUnit{\msg}{msg}',
    r'\DeclareSIUnit{\second}{s}'
]


TEX_LKN_DISS_PREAMBLE = [
    r'\usepackage{libertine}',
    r'\usepackage{mathptmx}',
    r'\usepackage[T1]{fontenc}',
    #r'\usepackage{bm}',
    r'\usepackage{amsmath}',
    #r'\usepackage{upgreek}',
    #r'\usepackage{textcomp}'
    r'\usepackage{siunitx}',
    r'\sisetup{range-units=single,'
    #r'        range-phrase = \,--\,,'
    #r'        list-units=single,'
    #r'        number-unit-product = \,,'
    #r'        separate-uncertainty = true,'
    #r'        multi-part-units = single,'
    #r'        detect-weight,'
    #r'        detect-display-math,'
    #r'        detect-inline-weight=math,'
    r'        per-mode=symbol,'
    r'        }',
    r'\DeclareSIUnit{\msg}{msg}',
    r'\DeclareSIUnit{\second}{s}'
]

PGF_DISS_PREAMBLE = [
    r'\usepackage{libertine}', # set later. check if it makes trouble.
    r'\usepackage{amsmath}',
    #r'\usepackage{bm}',
    #r'\renewcommand{\v}{\bm}',
    r'\usepackage{unicode-math}',  # unicode math setup
    #r'\setmathfont{XITS Math}',
    #r'\setmainfont{texgyretermes}',
    r'\usepackage{siunitx}',
    r'\sisetup{range-units=single,'
    #r'        range-phrase = \,--\,,'
    #r'        list-units=single,'
    #r'        number-unit-product = \,,'
    #r'        separate-uncertainty = true,'
    #r'        multi-part-units = single,'
    #r'        detect-weight,'
    #r'        detect-display-math,'
    #r'        detect-inline-weight=math,'
    r'        per-mode=symbol,'
    r'        }',
    r'\DeclareSIUnit{\msg}{msg}',
    r'\DeclareSIUnit{\second}{s}'
]

TEX_ACM_PREAMBLE = TEX_LKN_DISS_PREAMBLE
PGF_ACM_PREAMBLE = PGF_DISS_PREAMBLE

TEX_PREAMBLES = {
    DISS : TEX_LKN_DISS_PREAMBLE,
    IEEE : TEX_IEEE_PREAMBLE,
    ACM : TEX_ACM_PREAMBLE
}

PGF_PREAMBLES = {
    DISS : PGF_DISS_PREAMBLE,
    IEEE : PGF_IEEE_PREAMBLE,
    ACM : PGF_ACM_PREAMBLE
}

def texFigure(
        fig_width=None,
        fig_height=None,
        font_size=10,
        columns=1,
        line_width=1.0,
        axes_linewidth=1.0,
        legend_font_size=7,
        aspect_ratio="4to3",
        font_family='sans-serif',
        tex_preamble=IEEE,
        pgf_preamble=IEEE,
        verbose=False):
    """
    Replaces matplotlib's figure() call and uses Latex-friendly default values.

    Parameters
    -----------
    fig_width: float, optional
        Force a specific width of the figure.
    fig_height: float, optional
        Force a specific height of the figure. Note, if you do not set the height, it will be calculated according
        to your ratio. If you want a specific ratio, set both width and height
    font_size: int, optional
        Set font size to specific int
    columns: {1, 2}, optional
        Set one or two column style mode. How many columns does this figure use?
    line_width: float, optional, default 1.0
        Set line width of data
    axes_linewidth: float, optional, default 1.0
        Set line width of axes
    legend_font_size: int, optional, default 7
        Set font size of legend
    aspect_ratio: {4to3, golden}
        Set the aspect ratio of you figures. preferred are 4to3 or golden
    """

    if fig_width == None:
        fig_width = 3.487
        fig_width = columns * fig_width

    if fig_height is None:
        if aspect_ratio == 'golden':
            golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
            fig_height = fig_width * golden_mean  # height in inches

        if aspect_ratio == "4to3":
            golden_mean = 3.0 / 4.0  # Aesthetic ratio
            fig_height = fig_width * golden_mean  # height in inches

    params = {
        'axes.labelsize': font_size,  # fontsize for x and y labels (was 10)
        'axes.titlesize': font_size,
        'axes.linewidth': axes_linewidth,
        'font.size': font_size,  # was 10
        'font.family': font_family,
        'font.serif': ['Times'],
        'legend.fontsize': legend_font_size,  # was 10
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': [fig_width, fig_height],
        'lines.linewidth': line_width,  # 0.75
        'pdf.fonttype': 42,
        'text.usetex': True,
        'ps.useafm': True,
        'pdf.use14corefonts': True,
        'svg.fonttype' : 'none',
        'pgf.texsystem': 'xelatex',
        'pgf.rcfonts': False,
	     }

    if tex_preamble in TEX_PREAMBLES.keys():
        params['text.latex.preamble'] = TEX_PREAMBLES[tex_preamble]

    if tex_preamble == ACM:
        params['font.sans-serif'] = ['Linux Biolium']

    if pgf_preamble in PGF_PREAMBLES.keys():
        params['pgf.preamble'] = PGF_PREAMBLES[pgf_preamble]

    plt.rcParams.update(params)


def set_style_paper(fig_width, fig_height=None, font_size=8, aspect_ratio='golden'):
    texFigure(
        fig_width=fig_width,
        fig_height=fig_height,
        font_size=font_size,
        aspect_ratio=aspect_ratio,
        line_width=0.75,
        axes_linewidth=0.75,
        font_family='sans-serif',
        tex_preamble=DISS,
        pgf_preamble=DISS)

def save_figure(fig, figurefoldername, filename, rasterized=True, dpi=600, figure_format=['pdf'], verbose=False):
    """
    Saves figure in a folder with the given filename.

    Args:
        fig: the figure object
        figurefoldername: Where to store figure.
        filename: The filename without the file extension.
        rasterized: To rasterize to make smaller.
        dpi: Useful for pngs figures or pgf. Value should be > 300.
        figure_format: To be stored in ['pgf', 'pdf', 'png', ...]

    """

    for figure_format in figure_format:
        figurepath = figurefoldername + "/" + filename + "." + figure_format

        if verbose:
            print "Store figure in ", figurepath

        kwargs = {}

        if 'bbox_inches' not in kwargs:
            kwargs['bbox_inches'] = 'tight'

        if 'pad_inches' not in kwargs:
            kwargs['pad_inches'] = 0.01

        fig.savefig(figurepath, format=figure_format, dpi=dpi, **kwargs)




def filter_data(data, filters):
    # Apply filter
    for filter in filters:
        # Tuple: Range of values
        if isinstance(data_filter[filter], tuple):
            data = data.loc[(data[filter] >= data_filter[filter][0]) & (data[filter] <= data_filter[filter][1])]
        elif isinstance(data_filter[filter], list):
            data = data[data[filter].isin(data_filter[filter])]
        elif data_filter[filter] == None:
            data = data
        else:
            data = data.loc[data[filter] == data_filter[filter]]

    return data

figure_path = "./figures"
hdf_path = "./data/ovs_performance_cpu.hdf5"
data = pd.read_hdf(hdf_path)

if not os.path.exists(figure_path):
    os.makedirs(figure_path)


data_filter = {
    "megaflows_enabled": 1,
    "num_pkts": (2000, 5000),
    "burst_inter_time_mean": (1.5, 10.1)    
    
}

data = filter_data(data, data_filter)
        
print "Selected data has shape:", data.shape

x_fields = ["burst_inter_time_mean", "num_pkts"]
y_field = "cpu_ovs_sys"

set_style_paper(
    fig_width= (7.03 / 4.0) * 3,
    fig_height=(7.03 / 4.0) * 0.8 * 3,
    font_size=7 * 3
)

X1 = np.sort(data[x_fields[0]].unique())
X2 = np.sort(data[x_fields[1]].unique())
X1_grid, X2_grid = np.meshgrid(X1, X2)
X = np.array([X1_grid.ravel(), X2_grid.ravel()]).T

Y = []
for x_row in X:
    Y.append(data[(data[x_fields[0]] == x_row[0]) & (data[x_fields[1]] == x_row[1])][y_field].values[0])

Y_hm = np.array(Y).reshape(X2.shape[0], X1.shape[0])

fig, ax = plt.subplots(1)

cnt = plt.contourf(
    X1_grid, X2_grid, Y_hm, 30, vmin=0, vmax=np.max(Y), antialiased=False,
    cmap=plt.cm.get_cmap("magma")
)
for c in cnt.collections:
    c.set_edgecolor("face")
cbar = plt.colorbar()
plt.scatter(
    X[np.argmax(Y), 0], X[np.argmax(Y), 1], linewidth=2, facecolors='none', edgecolors='r', s=200, marker="o",
    cmap=plt.cm.get_cmap("plasma")
)
plt.xlim(X1.min() - X1.max() * (0.02), X1.max() + X1.max() * (0.02))
plt.ylim(X2.min() - X2.max() * (0.02), X2.max() + X2.max() * (0.02))
plt.xlabel("IAT [ms]")
ax.set_ylabel('Num. packets [1e3]')
plt.xticks(np.arange(2.0, 11.0, 1.0))
plt.yticks([2000, 3000, 4000, 5000])
ax.set_yticklabels([2, 3, 4, 5])
cbar.ax.get_yaxis().labelpad = 8 * 3
ax.get_yaxis().labelpad = 1
cbar.ax.set_ylabel('CPU time [s]', rotation=270)
plt.tight_layout()

save_figure(
    fig,
    figurefoldername=figure_path,
    filename='Heatmap_CPU'
)



x_fields = ["burst_inter_time_mean", "num_pkts"]
y_field = "latency"

X1 = np.sort(data[x_fields[0]].unique())
X2 = np.sort(data[x_fields[1]].unique())
X1_grid, X2_grid = np.meshgrid(X1, X2)
X = np.array([X1_grid.ravel(), X2_grid.ravel()]).T

Y = []
for x_row in X:
    Y.append(data[(data[x_fields[0]] == x_row[0]) & (data[x_fields[1]] == x_row[1])][y_field].values[0])

Y_hm = np.array(Y).reshape(X2.shape[0], X1.shape[0])

set_style_paper(
    fig_width=(7.03 / 4.0) * 3,
    fig_height=(7.03 / 4.0) * 0.8 * 3,
    font_size=7 * 3
)

fig, ax = plt.subplots(1)
cnt = plt.contourf(
    X1_grid, X2_grid, Y_hm, 30, vmin=np.min(Y), vmax=np.max(Y), antialiased=False,
    cmap=plt.cm.get_cmap("magma")
)
for c in cnt.collections:
    c.set_edgecolor("face")
cbar = plt.colorbar(
    ticks=[0.5, 0.6, 0.7, 0.8, 0.9]
)
plt.scatter(
    X[np.argmax(Y), 0], X[np.argmax(Y), 1], linewidth=2, facecolors='none', edgecolors='r', s=200, marker="o",
    cmap=plt.cm.get_cmap("plasma")
)
plt.xlim(X1.min() - X1.max() * (0.02), X1.max() + X1.max() * (0.02))
plt.ylim(X2.min() - X2.max() * (0.02), X2.max() + X2.max() * (0.02))
plt.xlabel("IAT [ms]")
plt.ylabel("\# packets")
plt.xticks(np.arange(2.0, 11.0, 1.0))
plt.yticks([2000, 3000, 4000, 5000])
ax.set_yticklabels([2, 3, 4, 5])
ax.set_ylabel('Num. packets [1e3]')
cbar.ax.get_yaxis().labelpad = 8 * 3
cbar.ax.set_ylabel('Latency [ms]', rotation=270)
ax.get_yaxis().labelpad = 1.5
plt.tight_layout()

save_figure(
    fig,
    figurefoldername=figure_path,
    filename='Heatmap_Latency'
)

