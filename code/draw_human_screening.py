import numpy as np
import matplotlib.pyplot as plt

import pylab as mpl

# mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


category_names = ['Results predicted by GBDT',
                  'Results of inclusion by MV',
                  'Accuracy']

results = {
    'vote ≥ 3': [	197,	167,	84.77],
    'vote = 2': [	152,	22,	14.47],
    'vote = 1': [	611,	25,	4.09],
    'vote = 0': [	500,	0,	0],
}


font = {

    # 'family' : 'Arial',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 10.5,
        }

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    # category_colors = plt.get_cmap('RdYlGn')(
    #     np.linspace(0.15, 0.85, data.shape[1]))
    category_colors = [ "#63BBD0","#F4B183","#6BB398"]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        # r, g, b, _ = color
        # text_color = 'black' if r * g * b < 0.5 else 'darkgrey'
        text_color = 'black'
        if i != 2:
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                ax.text(x, y, str(int(c)), ha='center', va='center',
                        color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=8.5)
    # ax.legend(ncol=len(category_names), bbox_to_anchor=(1, 0), fontsize=10.5)

    return fig, ax


fig,ax = survey(results, category_names)
plt.yticks(fontsize=10.5)
plt.savefig("pic/human-screening.png",dpi=500,bbox_inches='tight')#解决图片不清晰，不完整的问题
plt.show()
