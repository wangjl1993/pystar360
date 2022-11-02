from enum import Enum
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pystar360.utilities.misc import threadingDecorator
from pystar360.utilities.helper import *


class palette(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    PURPLE = (125, 38, 205)
    YELLOW = (255, 255, 102)


################################################################################
#### plot bboxes on image
################################################################################


def plt_bboxes_on_img(
    bboxes,
    img,
    img_h,
    img_w,
    startline,
    axis=1,
    vis_lv=1,
    resize_ratio=0.1,
    default_color=palette.GREEN,
    use_3drect=False,
):
    if not bboxes:
        return img

    img = resize_img(img, resize_ratio)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_h, img_w = get_img_size2(img_h, img_w, resize_ratio)
    for b in bboxes:

        if b.is_defect != 0 or b.is_3ddefect != 0:
            color = palette.RED
        else:
            color = default_color

        if use_3drect:
            points = frame2rect(b.curr_rect3d, startline, img_h, img_w, axis=axis)
        else:
            points = frame2rect(b.curr_rect, startline, img_h, img_w, axis=axis)
        cv2.rectangle(img, points[0], points[1], color.value, 1)

        if vis_lv < 1:
            text = b.name
        elif vis_lv < 2:
            text = concat_str(b.name, b.index)
        elif vis_lv < 3:
            text = concat_str(b.name, b.index, b.conf_score)
        else:
            text = concat_str(b.name, b.index, b.conf_score)
            if use_3drect:
                proposal_points = frame2rect(b.proposal_rect3d, startline, img_h, img_w, axis=axis)
            else:
                proposal_points = frame2rect(b.proposal_rect, startline, img_h, img_w, axis=axis)
            cv2.rectangle(img, proposal_points[0], proposal_points[1], palette.PURPLE.value, 1)

        cv2.putText(img, text, points[0], cv2.FONT_HERSHEY_PLAIN, 1.5, color.value)

    return img


@threadingDecorator
def plt_bboxes_on_img_threading(
    save_path, bboxes, img, img_h, img_w, startline, axis=1, vis_lv=1, resize_ratio=0.1, default_color=palette.GREEN
):
    img = plt_bboxes_on_img(
        bboxes,
        img,
        img_h,
        img_w,
        startline,
        axis=axis,
        vis_lv=vis_lv,
        resize_ratio=resize_ratio,
        default_color=default_color,
    )
    cv2.imwrite(str(save_path), img)


################################################################################
#### plot heatmap
################################################################################


def heatmap(data, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        # fig, ax = plt.subplots()
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plt_heatmap(data, save_path, annotated=True):
    fig, ax = plt.subplots()
    im, _ = heatmap(data, ax=ax, cmap="magma_r")
    if annotated:
        _ = annotate_heatmap(im, valfmt="{x:.1f}")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=250)
