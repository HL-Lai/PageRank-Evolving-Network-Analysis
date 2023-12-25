# PageRank: Evolving Network Analysis

This repository contains an implementation of a network analysis project based on evolving graphs. The project explores the concept of PageRank in a dynamic context.

## Dataset

The dataset used in this project is derived from the Autonomous Systems AS-733 from the University of Oregon Route Views Project. The dataset spans from November 8, 1997, to January 2, 2000, encompassing a total of 733 daily instances. The dataset is publicly accessible [here](http://snap.stanford.edu/data/as.html).

## Implementation

The project uses two parts of the graph: the initial graph and the evolving graph. The graph on November 8, 1997, is set as the initial graph, with the assumption that the edges are changing daily.

## Experiment

The experiment involves a comparison between different probing methods: **Random Probing**, **Round-Robin Probing**, **Proportional Probing**, and **Priority Probing**.

## References

Bahmani, B., Kumar, R., Mahdian, M., & Upfal, E. (2012). PageRank on an evolving graph. *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. https://doi.org/10.1145/2339530.2339539

