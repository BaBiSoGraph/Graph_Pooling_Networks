import tensorflow as tf


@tf.function
def maxPartitionPooling(tensor, partitions: list):
    """
    Performs partition pooling on the input.

    Each entry in `output` is the max of the corresponding partition.
    The data is assumed to be 1D (excluding feature dimensions). If you want
    to use this on 2, 3 or higher dimesional data use flattening in the previous layer.

    Note:
        using this function in a neural network requires to disable tensorflow eager execution

    Args:
        input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
            Shape `[batch, nodes, features]` tensor to pool over.
        partitions: A list of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition. The sublists have to be of equal length.

      Returns:
        A `Tensor`. Has the same type as `input`.
    """
    return tf.math.reduce_max(tf.gather(tensor,
                                        tf.constant(partitions),
                                        batch_dims=0,
                                        axis=1),
                              axis=2)


@tf.function
def sumPartitionPooling(tensor, nodeWeights, partitions: list):
    """
    Performs partition pooling on the input.

    Each entry in `output` is the (weighted) sum of the corresponding partition.
    The data is assumed to be 1D (excluding feature dimensions). If you want
    to use this on 2, 3 or higher dimesional data use flattening in the previous layer.

    Note:
        Passing 'nodeWeights' is NOT optional. For unique nodes in
        the partition the default weight is 1.

    Args:
        tensor: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
            Shape `[batch, nodes, features]` tensor to pool over.
        nodeWeights: A 2D `Tensor` (tf.constant). Shape same as `partitions`. Contains the weights the nodes are weighted
            to correct for nodes appearing multiple times in a pooled node due to padding.
            Thus the weights of one partitions sublist should sum up to the number of pooled nodes.
        partitions: A list of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition. The sublists have to be of equal length.

      Returns:
        A `Tensor`. Has the same type as `input`.
    """
    product = tf.math.multiply(tf.gather(tensor,
                                         tf.constant(partitions),
                                         batch_dims=0,
                                         axis=1),
                               tf.reshape(nodeWeights, [1, *nodeWeights.shape, 1]))

    return tf.math.reduce_sum(product, axis=2)


@tf.function
def averagePartitionPooling(tensor, nodeWeights, partitionSizes, partitions: list):
    """
    Performs partition pooling on the input.

    Each entry in `output` is the (weighted) average of the corresponding partition.
    The data is assumed to be 1D (excluding feature dimensions). If you want
    to use this on 2, 3 or higher dimesional data use flattening in the previous layer.

    Note:
        Passing 'nodeWeights' and 'partitionSizes' is NOT optional. For unique nodes in
        the partition the default weight is 1.

    Args:
        tensor: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
            Shape `[batch, nodes, features]` tensor to pool over.
        nodeWeights: A 2D `Tensor` (tf.constant). Shape same as `partitions`. Contains the weights the nodes are weighted
            to correct for nodes appearing multiple times in a pooled node due to padding.
            Thus the weights of one partitions sublist should sum up to the number of pooled nodes.
        partitionSizes: A 1D `Tensor` (tf.constant). Length equal to number of partitions.
        partitions: A list of lists. Each sublist contains `ints`. The indices
            of the nodes that comprise one partition. The sublists have to be of equal length.

      Returns:
        A `Tensor`. Has the same type as `input`.
    """
    product = tf.math.multiply(tf.gather(tensor,
                                         tf.constant(partitions),
                                         batch_dims=0,
                                         axis=1),
                               tf.reshape(nodeWeights, [1, *nodeWeights.shape, 1]))

    return tf.math.divide(tf.math.reduce_sum(product, axis=2),
                          tf.reshape(partitionSizes, [1, *partitionSizes.shape, 1]))


if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")
