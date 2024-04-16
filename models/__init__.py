from .deformable_detr import build
from .deformable_detrtrack_test import build as build_tracktest
from .deformable_detrtrack_train import build as build_tracktrain
from .trackernew import Tracker
from .save_track import save_track


def build_model(args):
    return build(args)

def build_tracktrain_model(args):
    return build_tracktrain(args)

def build_tracktest_model(args):
    return build_tracktest(args)
