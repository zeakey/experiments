import visdom
import numpy as np

vis = visdom.Visdom(port=8098)

x = np.array([np.arange(0, 10, 0.01), np.arange(0, 10, 0.01)]).transpose()

y = np.sin(x)
y[:, 1] = np.cos(x[:, 1])

vis.line(y, x,
    opts=dict({
        "title": "aa",
        "fillarea": True,
        "ytickmax": 1,
        "ytickmin": -1,
        "xtickmax": 9,
        "xtickmin": 0,
        "xtickstep":
1}), win=4)
