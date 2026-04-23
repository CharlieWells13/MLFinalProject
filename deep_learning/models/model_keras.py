import tensorflow as tf
from tensorflow.keras import Model, layers


def _residual_block(x: tf.Tensor, filters: int, stride: int = 1, name: str = "block") -> tf.Tensor:
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, kernel_size=1, strides=stride, padding="same", use_bias=False, name=f"{name}_proj_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_model(
    input_shape: tuple = (224, 224, 3),
    dropout: float = 0.2,
    apply_sigmoid: bool = False,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> Model:
    if pretrained:
        raise ValueError("Pretrained weights are not provided for this custom Keras ResNet-18 implementation.")

    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)

    x = _residual_block(x, 64, stride=1, name="layer1_block1")
    x = _residual_block(x, 64, stride=1, name="layer1_block2")

    x = _residual_block(x, 128, stride=2, name="layer2_block1")
    x = _residual_block(x, 128, stride=1, name="layer2_block2")

    x = _residual_block(x, 256, stride=2, name="layer3_block1")
    x = _residual_block(x, 256, stride=1, name="layer3_block2")

    x = _residual_block(x, 512, stride=2, name="layer4_block1")
    x = _residual_block(x, 512, stride=1, name="layer4_block2")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(4, activation="sigmoid" if apply_sigmoid else None, name="bbox")(x)

    model = Model(inputs=inputs, outputs=outputs, name="resnet18_bbox_regressor")

    if freeze_backbone:
        for layer in model.layers:
            if not layer.name.startswith("head_") and layer.name != "bbox":
                layer.trainable = False

    return model
