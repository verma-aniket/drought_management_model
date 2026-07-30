

def plot_format(fig, xlab, ylab, issubplot=True, majcol='#D5D8DC', mincol='#EAECEE', grid=True):
    if issubplot == True:
        fig.set(xlabel=xlab, ylabel=ylab)
    else:
        fig.xlabel(xlab)
        fig.ylabel(ylab)

    if grid == True:
        fig.grid(zorder=0)
        fig.grid(visible=True, which="major", color=majcol, linestyle="-")
        fig.grid(visible=True, which="minor", color=mincol, linestyle="-")
        fig.minorticks_on()

    fig.tick_params(axis="y", which="both", direction="in", length=0)
    fig.tick_params(axis="x", which="both", direction="in", length=0)
    return fig