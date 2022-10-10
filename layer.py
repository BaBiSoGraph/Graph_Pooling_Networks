import tensorflow as tf
import numpy as np
from algorithm import maxPartitionPooling,\
                      sumPartitionPooling,\
                      averagePartitionPooling


class PartitionPooling(tf.keras.layers.Layer):
    """
    Partition Pooling layer base class

    This is the base class for partition pooling layers.
    A partition list is taken and adjusted, such that each sublist has
    equal length. Weights to correct for multiple entries are constructed
    additionally.

    Args:
        partitions: List of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition.
    """

    def __init__(self, partitions: list, **kwargs):
        super(PartitionPooling, self).__init__(**kwargs)
        self.partitions = partitions
        self.adjustedPartitions, self.countWeights = self.adjustPartitionLength(self.partitions)

    def adjustPartitionLength(self, part):
        """
        Adjusts the length of each sublist of `part` to the length of the
            largest sublist. Done by padding with the last
            element of the respective sublist.
        Args:
            part: A list of lists. Each sublist contains `ints`. The indices
                  of the nodes that comprise one partition.
        Return:
            adjustedPartitions: list. Length adjusted partitions
            countWeights: list. Weightes according to number of multiple entries.
        """

        l = [len(e) for e in part]  # List with lengths of sublists in part
        partLen = len(part)
        maxLen = max(l)  # maximum length

        adjustedPartitions = np.zeros((partLen, maxLen), dtype=int)
        countWeights = np.ones((partLen, maxLen))

        # Fill each sublist with its last element up to lenght 'maxLen'
        for i, e in enumerate(part):
            lenDiff = maxLen - l[i]
            adjustedPartitions[i] = np.pad(e, (0, lenDiff), 'edge')
            countWeights[i, -(lenDiff + 1):] = 1. / (lenDiff + 1)

        return adjustedPartitions.tolist(), countWeights.tolist()

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "partitions": self.partitions}


class MaxPartitionPooling(PartitionPooling):
    """
    Max Partition Pooling layer.

    This layer pools the input data into a new tensor via max pooling
    with respect to the desired `partitions`.
    The data is assumed to be 1D (excluding feature dimensions). If you want
    to use this on 2, 3 or higher dimesional data use flattening in the previous layer.
    The corresponding output dimension will be `(None, len(partitions), features)`.

    Args:
        partitions: A list of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition.


    Input shape:
        3+D tensor with shape: `batch_shape + (nodes, features)`

    Output shape:
        3+D tensor with shape: `batch_shape + (len(partitions), features)`

    Returns:
        A tensor of rank 3
    """

    def __init__(self, partitions: list, **kwargs):
        super(MaxPartitionPooling, self).__init__(partitions, **kwargs)

    def call(self, layer):
        return maxPartitionPooling(layer, self.adjustedPartitions)


class SumPartitionPooling(PartitionPooling):
    """
    Sum Partition Pooling layer.

    This layer pools the input data onto a new tensor via sum pooling
    with respect to the desired `partitions`.
    The data is assumed to be 1D (excluding feature dimensions). If you want
    to use this on 2, 3 or higher dimesional data use flattening in the previous layer.
    The corresponding output dimension will be `(None, len(partitions), features)`.

    Args:
        partitions: A list of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition.


    Input shape:
        3+D tensor with shape: `batch_shape + (nodes, features)`

    Output shape:
        3+D tensor with shape: `batch_shape + (len(partitions), features)`

    Returns:
        A tensor of rank 3
    """

    def __init__(self, partitions: list, **kwargs):
        super(SumPartitionPooling, self).__init__(partitions, **kwargs)
        self.nodeWeights = tf.constant(self.countWeights, dtype="float32")

    def call(self, layer):
        return sumPartitionPooling(layer, self.nodeWeights, self.adjustedPartitions)


class AveragePartitionPooling(PartitionPooling):
    """
    Average Partition Pooling layer.

    This layer pools the input data onto a new tensor via average pooling
    with respect to the desired `partitions`.
    The data is assumed to be 1D (excluding feature dimensions). If you want
    to use this on 2, 3 or higher dimesional data use flattening in the previous layer.
    The corresponding output dimension will be `(None, len(partitions), features)`.

    Args:
        partitions: A list of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition.


    Input shape:
        3+D tensor with shape: `batch_shape + (nodes, features)`

    Output shape:
        3+D tensor with shape: `batch_shape + (len(partitions), features)`

    Returns:
        A tensor of rank 3
    """

    def __init__(self, partitions: list, **kwargs):
        super(AveragePartitionPooling, self).__init__(partitions, **kwargs)
        self.nodeWeight = tf.constant(self.countWeights, dtype="float32")
        self.partitionSizes = tf.constant(np.sum(self.countWeights, axis=-1), dtype="float32")

    def call(self, layer):
        return averagePartitionPooling(layer, self.nodeWeight, self.partitionSizes, self.adjustedPartitions)


if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")
