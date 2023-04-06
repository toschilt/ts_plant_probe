"""
Implements basic color returns for clustering.
"""

import matplotlib.colors as mcolors

def get_color_from_cluster(cluster):
    """
    Returns a string containing a different color for each cluster number.

    Colors are returned in hexadecimal format.
    """
    return list(mcolors._colors_full_map.values())[cluster + 10]