import cv2
import matplotlib.pyplot as plt
import torch
from collections import namedtuple
import numpy as np

"""###############"""
"""# Definitions #"""
"""###############"""
# following definition are copied from github repository:
#   mcordts/cityscapesScripts/cityscapesscripts/helpers/labels.py
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name          id    trainId   category   catId     hasInstances   ignoreInEval   color
    Label(  'background', 0,      0,    'void'     , 0,        False,         False,          (  255,  0,  0) ),
    Label(  'face',       1,      1,    'void'     , 0,        False,         False,          (  0,  255,  0) ),
    Label(  'hair',       2,      2,    'void'      ,0,        False,         False,          (  0,  0,  255) ),
]

"""###################"""
"""# Transformations #"""
"""###################"""

def logits2trainId(logits):
    """
    Transform output of network into trainId map
    :param logits: output tensor of network, before softmax, should be in shape (#classes, h, w)
    """
    # squeeze logits
    # num_classes = logits.size[1]
    upsample = torch.nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=False)
    logits = upsample(logits.unsqueeze_(0))
    logits.squeeze_(0)
    logits = torch.argmax(logits, dim=0)

    return logits


def trainId2color(train_dir, id_map, name):
    """
    Transform trainId map into color map
    :param train_dir: the path to the training directory
    :param id_map: torch tensor
    :param name: name of image, eg. 'gtFine/test/leverkusen/leverkusen_000027_000019_gtFine_labelTrainIds.png'
    """
    # transform = {label.trainId: label.color for label in labels}
    assert len(id_map.shape) == 2, 'Id_map must be a 2-D tensor of shape (h, w) where h, w = H, W / output_stride'
    h, w = id_map.shape
    color_map = np.zeros((h, w, 3))
    id_map = id_map.cpu().numpy()
    for label in labels:
        if not label.ignoreInEval:
            color_map[id_map == label.trainId] = np.array(label.color)
    color_map = color_map.astype(np.uint8)
    # color_map = cv2.resize(color_map, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)

    # save trainIds and color
    #cv2.imwrite(train_dir + '/' + name, id_map)
    #name = name.replace('labelTrainIds', 'color')
    cv2.imwrite(train_dir + '/' + name, color_map)

    return color_map


def trainId2LabelId(train_dir, train_id, name):
    """
        Transform trainId map into labelId map
        :param train_dir: the path to the training directory
        :param id_map: torch tensor
        :param name: name of image, eg. 'gtFine/test/leverkusen/leverkusen_000027_000019_gtFine_labelTrainIds.png'
        """
    assert len(train_id.shape) == 2, 'Id_map must be a 2-D tensor of shape (h, w) where h, w = H, W / output_stride'
    h, w = train_id.shape
    label_id = np.zeros((h, w, 3))
    train_id = train_id.cpu().numpy()
    for label in labels:
        if not label.ignoreInEval:
            label_id[train_id == label.trainId] = np.array([label.id]*3)
    label_id = label_id.astype(np.uint8)
    # label_id = cv2.resize(label_id, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)

    name = name.replace('labelTrainIds', 'labelIds')
    cv2.imwrite(train_dir + '/' + name, label_id)


if __name__ == '__main__':
    pass
    # trainId = cv2.imread('/media/ubuntu/disk/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png')
    # trainId2color(trainId[:, :, 0])
