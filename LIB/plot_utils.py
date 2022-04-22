import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
import os
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

plt.close("all")

def show_animation(q, Xgrid=None, cycles=1, frequency = 1, figure_number = None):
    ntime = q.shape[-1]

    if figure_number == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(num=figure_number)

    if len(Xgrid) == 1:
        line, = ax.plot(Xgrid[0], q[..., 0])
        plt.ylim([np.min(q), np.max(q)])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q(x,t)$")
        for t in range(0, cycles * ntime, frequency):
            line.set_data(Xgrid[0],q[..., t % ntime].ravel())
            # fig.savefig("../imgs/FTR_6modes_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)


    else:

        h = ax.pcolormesh(Xgrid[0], Xgrid[1], q[..., 0])
        h.set_clim(np.min(q), np.max(q))
        ax.axis("image")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        for t in range(0, cycles * ntime, frequency):
            h.set_array(q[:-1, :-1, t % ntime].ravel())
            # fig.savefig("../imgs/FTR_6modes_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)


def bound_exceed_cmap(Index_upper=10, Index_lower=10):
    from matplotlib.colors import ListedColormap
    viridis = plt.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    Ir = Index_lower
    Ib = Index_upper
    reds = np.ones([Ir, 4])
    blacks = np.ones([Ib, 4])
    for i in range(Ir):
        reds[i, :] = [1 - i / Ir, 0, 0, 1]
    for i in range(Ib):
        blacks[i, :] = [i / Ib, i / Ib, i / Ib, 1]
    newcolors = np.concatenate([reds, newcolors, blacks])
    return ListedColormap(newcolors)


def save_fig(filepath, figure=None, **kwargs ):
    import tikzplotlib
    import os

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    tikzplotlib.save(
        figure = figure,
        filepath=fpath+".tex",
        axis_height = '\\figureheight',
        axis_width = '\\figurewidth',
        override_externals = True,
        **kwargs
    )
    plt.tight_layout()
    figure.savefig(fpath + ".pdf", dpi=600, transparent=True)

def autolabel(rects, ax, fmt ):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')