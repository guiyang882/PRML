# -*- coding: utf-8 -*-

import trees
import treePlotter

filename = "data.csv"
mydat,mylabels = trees.createDataSet(filename)
myTree = trees.createTree(mydat,mylabels)
treePlotter.createPlot(myTree)
