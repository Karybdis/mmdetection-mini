from .voc import VOCDataset
from .builder import DATASETS

@DATASETS.register_module()
class VOC_RemoteSense(VOCDataset):

    CLASSES = ("person", "vehicle")

    def __init__(self,**kwargs):
        super(VOC_RemoteSense, self).__init__(**kwargs)

