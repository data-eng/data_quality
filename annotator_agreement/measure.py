import numpy

def inter_annotator_agreement( npz_object ):
    # Inputs:
    # a 2D numpy array where each row is multiple annotations
    # of the same object and the last time is the ground truth.
    # A label of -1 means "very bad sample, not possible to annotate"
    #
    # Returns an array with:
    # 1: perfect agreement (or as good as it gets on this task)
    # 0: bad agreement
    # or a fraction inbetween

    retv = []
    for row in npz_object["y"]:
        t = row[3]
        l = row[0:3]
        if t == -1: retv.append( 0.0 )
        else:
            diffs = l - t
            sqerr = (diffs*diffs).sum()
            if sqerr == 0: retv.append( 1.0 )
            elif sqerr == 1: retv.append( 0.5 )
            else: retv.append( 0.2 )
    assert npz_object["y"].shape[0] == len(retv)
    return numpy.array( retv )


