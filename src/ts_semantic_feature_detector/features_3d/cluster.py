"""
Encapsules crop clusters through time.
"""

class Cluster:
    """
    Abstracts a crop cluster.
    
    Attributes:
        id (int): the cluster id.
        age (int): the cluster age.
        num_crops (int): the number of crops associated to this cluster.
    """ 

    # The ids of the deleted clusters.
    next_cluster_id = 0

    def __init__(
        self,
        id: int
    ):
        """
        Initializes the crop cluster.

        Args:
            id (int): the cluster id.
        """
        
        self.id = id
        self.age = 0
        self.num_crops = 0