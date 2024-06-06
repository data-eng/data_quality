import numpy

def inter_annotator_agreement(npz_object):
    """
    Compute the inter-annotator agreement.

    :param npz_object: numpy array (n x k) with annotations where:
        - n is the number of samples
        - k-1 represents annotations from k-1 annotators
        - the k-th element represents the ground truth label
        - a label of -1 indicates a very bad sample (not possible to annotate)
    :return: numpy array (n x 1) with agreement scores for each sample:
        - s = 1: perfect agreement (or as good as it gets on this task)
        - s = 0: bad agreement
        - 0 < s < 1: partial agreement
    """
    scores = []

    for sample in npz_object["y"]:
        ground_truth = sample[3]
        labels = sample[0:3]

        if ground_truth == -1: scores.append(0.0)
        else:
            diffs = labels - ground_truth
            sqerr = (diffs*diffs).sum()

            if sqerr == 0: scores.append(1.0)
            elif sqerr == 1: scores.append(0.5)
            else: scores.append(0.2)
            
    assert npz_object["y"].shape[0] == len(scores)
    return numpy.array(scores)