from . import measure

def quality(npz_object):
    return measure.inter_annotator_agreement(npz_object)