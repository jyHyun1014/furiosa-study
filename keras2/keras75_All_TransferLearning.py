# Keras Applications에서 제공하는 다양한 사전학습 CNN 모델(VGG, ResNet, DenseNet, MobileNet 등)을 불러와 전체 가중치와 훈련 가능한 가중치의 개수를 비교

import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [
    VGG16(), VGG19(),
    ResNet50(), ResNet50V2(),
    ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
    DenseNet201(), DenseNet169(), DenseNet121(),
    InceptionV3(), InceptionResNetV2(),
    MobileNet(), MobileNetV2(),
    MobileNetV3Small(), MobileNetV3Large(),
    NASNetMobile(), NASNetLarge(),
    EfficientNetB0(), EfficientNetB1(), EfficientNetB2(),
]

# model = Xception()

# model.trainable = True
# model.summary()

# print("======================================")
# print("모델명 :", model.name)
# print("전체 가중치 개수 :", len(model.weights))
# print("전체 훈령가능 가중치 개수:", len(model.trainable_weights))

for model in model_list:
    model.trainable = True
    print("======================================")
    print("모델명 :", model.name)
    print("전체 가중치 개수 :", len(model.weights))
    print("전체 훈련가능 가중치 개수:", len(model.trainable_weights))

# ======================================
# 모델명 : vgg16
# 전체 가중치 개수 : 32
# 전체 훈련가능 가중치 개수: 32
# ======================================
# 모델명 : vgg19
# 전체 가중치 개수 : 38
# 전체 훈련가능 가중치 개수: 38
# ======================================
# 모델명 : resnet50
# 전체 가중치 개수 : 320
# 전체 훈련가능 가중치 개수: 214
# ======================================
# 모델명 : resnet50v2
# 전체 가중치 개수 : 272
# 전체 훈련가능 가중치 개수: 174
# ======================================
# 모델명 : resnet101
# 전체 가중치 개수 : 626
# 전체 훈련가능 가중치 개수: 418
# ======================================
# 모델명 : resnet101v2
# 전체 가중치 개수 : 544
# 전체 훈련가능 가중치 개수: 344
# ======================================
# 모델명 : resnet152
# 전체 가중치 개수 : 932
# 전체 훈련가능 가중치 개수: 622
# ======================================
# 모델명 : resnet152v2
# 전체 가중치 개수 : 816
# 전체 훈련가능 가중치 개수: 514
# ======================================
# 모델명 : densenet201
# 전체 가중치 개수 : 1006
# 전체 훈련가능 가중치 개수: 604
# ======================================
# 모델명 : densenet169
# 전체 가중치 개수 : 846
# 전체 훈련가능 가중치 개수: 508
# ======================================
# 모델명 : densenet121
# 전체 가중치 개수 : 606
# 전체 훈련가능 가중치 개수: 364
# ======================================
# 모델명 : inception_v3
# 전체 가중치 개수 : 378
# 전체 훈련가능 가중치 개수: 190
# ======================================
# 모델명 : inception_resnet_v2
# 전체 가중치 개수 : 898
# 전체 훈련가능 가중치 개수: 490
# ======================================
# 모델명 : mobilenet_1.00_224
# 전체 가중치 개수 : 137
# 전체 훈련가능 가중치 개수: 83
# ======================================
# 모델명 : mobilenetv2_1.00_224
# 전체 가중치 개수 : 262
# 전체 훈련가능 가중치 개수: 158
# ======================================
# 모델명 : MobileNetV3Small
# 전체 가중치 개수 : 210
# 전체 훈련가능 가중치 개수: 142
# ======================================
# 모델명 : MobileNetV3Large
# 전체 가중치 개수 : 266
# 전체 훈련가능 가중치 개수: 174
# ======================================
# 모델명 : nasnet_mobile
# 전체 가중치 개수 : 1126
# 전체 훈련가능 가중치 개수: 742
# ======================================
# 모델명 : nasnet_large
# 전체 가중치 개수 : 1546
# 전체 훈련가능 가중치 개수: 1018
# ======================================
# 모델명 : efficientnetb0
# 전체 가중치 개수 : 314
# 전체 훈련가능 가중치 개수: 213
# ======================================
# 모델명 : efficientnetb1
# 전체 가중치 개수 : 442
# 전체 훈련가능 가중치 개수: 301
# ======================================
# 모델명 : efficientnetb2
# 전체 가중치 개수 : 442
# 전체 훈련가능 가중치 개수: 301