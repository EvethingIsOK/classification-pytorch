from .mobilenet_v1 import mobilenet_v1
from .mobilenet_v2 import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16
from .vit import vit
from .ghostnet import ghostnet

get_model_from_name = {
    "ghostnet"         : ghostnet,
    "mobilenet_v1"     : mobilenet_v1,
    "mobilenet_v2"     : mobilenet_v2,
    "resnet50"      : resnet50,
    "vgg16"         : vgg16,
    "vit"           : vit
}