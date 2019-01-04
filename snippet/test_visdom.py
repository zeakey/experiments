import visdom
import numpy as np

vis = visdom.Visdom(port=8100)

y = np.random.rand(10, 2)

vis.line(y, opts=dict({"legend": ["a", "b"], "title": "aa", "fillarea": True}))