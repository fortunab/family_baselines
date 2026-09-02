"""
TensorFlow 2.x Foundation Model Factory for Colorectal Histology.
Supports 4 Performant Vision & Pathology Foundation Models:
1. Meta ConvNeXt-Large / Base (tf.keras.applications.ConvNeXtLarge)
2. Google EfficientNetV2-L (tf.keras.applications.EfficientNetV2L)
3. Google Vision Transformer ViT-Base (google/vit-base-patch16-224-in21k via TFAutoModel / Keras)
4. Google Big Transfer BiT / ResNet152V2 (tf.keras.applications.ResNet152V2)
"""

import os
from typing import Tuple, Dict, Any, Optional
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_convnext_foundation(
    num_classes: int = 8,
    variant: str = "large",
    drop_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    input_shape = (224, 224, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    if variant.lower() == "large":
        base_model = tf.keras.applications.ConvNeXtLarge(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        model_name = "convnext_large"
    elif variant.lower() == "small":
        base_model = tf.keras.applications.ConvNeXtSmall(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        model_name = "convnext_small"
    else:
        base_model = tf.keras.applications.ConvNeXtBase(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        model_name = "convnext_base"

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
    x = layers.Dropout(drop_rate, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="histology_classifier"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"tf_{model_name}_foundation")
    config = {
        "model_name": model_name,
        "img_size": 224,
        "num_classes": num_classes,
        "base_model": base_model
    }
    return model, config


def build_efficientnetv2_foundation(
    num_classes: int = 8,
    variant: str = "l",
    drop_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    input_shape = (224, 224, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    if variant.lower() == "l" or variant.lower() == "large":
        base_model = tf.keras.applications.EfficientNetV2L(
            include_top=False, weights="imagenet-21k", input_shape=input_shape
        )
        model_name = "efficientnetv2_l"
    elif variant.lower() == "s":
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False, weights="imagenet-21k", input_shape=input_shape
        )
        model_name = "efficientnetv2_s"
    else:
        base_model = tf.keras.applications.EfficientNetV2M(
            include_top=False, weights="imagenet-21k", input_shape=input_shape
        )
        model_name = "efficientnetv2_m"

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(drop_rate, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="histology_classifier"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"tf_{model_name}_foundation")
    config = {
        "model_name": model_name,
        "img_size": 224,
        "num_classes": num_classes,
        "base_model": base_model
    }
    return model, config


def build_vit_foundation(
    num_classes: int = 8,
    hf_model_id: str = "google/vit-base-patch16-224-in21k",
    drop_rate: float = 0.2,
    l2_reg: float = 1e-4
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    input_shape = (224, 224, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    try:
        from transformers import TFAutoModel
        print(f"[TF-ViT] Loading HuggingFace TFAutoModel: {hf_model_id}...")
        # Channels first transpose for HuggingFace ViT [B, H, W, C] -> [B, C, H, W]
        x_trans = layers.Permute((3, 1, 2), name="channel_first_permute")(inputs)
        vit_backbone = TFAutoModel.from_pretrained(hf_model_id)
        vit_out = vit_backbone(pixel_values=x_trans)[0]  # [B, seq_len, hidden_size]
        cls_token = vit_out[:, 0, :]  # CLS token representation
        x = layers.LayerNormalization(epsilon=1e-6, name="cls_norm")(cls_token)
        x = layers.Dropout(drop_rate, name="head_dropout")(x)
        outputs = layers.Dense(
            num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2_reg), name="histology_classifier"
        )(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="tf_vit_foundation")
        config = {"model_name": "vit_base_patch16", "img_size": 224, "num_classes": num_classes, "base_model": vit_backbone}
        return model, config
    except Exception as e:
        print(f"[TF-ViT] HuggingFace TFViT load notice ({e}), building pure Keras ViT backbone...")

    # Native Keras Vision Transformer implementation fallback
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    projection_dim = 768
    num_heads = 12
    transformer_layers = 12

    # Patch extraction & linear projection
    x = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid", name="patch_projection")(inputs)
    x = layers.Reshape((num_patches, projection_dim))(x)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + pos_embed

    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attn, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp = layers.Dense(projection_dim * 4, activation=tf.nn.gelu)(x3)
        mlp = layers.Dropout(0.1)(mlp)
        mlp = layers.Dense(projection_dim)(mlp)
        mlp = layers.Dropout(0.1)(mlp)
        x = layers.Add()([mlp, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(drop_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2_reg), name="histology_classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="tf_vit_keras_foundation")
    config = {"model_name": "vit_base_keras", "img_size": 224, "num_classes": num_classes, "base_model": None}
    return model, config


def build_bit_foundation(
    num_classes: int = 8,
    drop_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    input_shape = (224, 224, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    base_model = tf.keras.applications.ResNet152V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    model_name = "bit_resnet152v2"

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(drop_rate, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="histology_classifier"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"tf_{model_name}_foundation")
    config = {
        "model_name": model_name,
        "img_size": 224,
        "num_classes": num_classes,
        "base_model": base_model
    }
    return model, config


def get_tf_foundation_model(model_name: str, num_classes: int = 8) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    name_lower = model_name.lower().replace("-", "_")
    if "convnext" in name_lower:
        variant = "large" if "large" in name_lower else ("small" if "small" in name_lower else "base")
        return build_convnext_foundation(num_classes=num_classes, variant=variant)
    elif "efficientnet" in name_lower or "effnet" in name_lower:
        variant = "l" if "l" in name_lower else ("s" if "s" in name_lower else "m")
        return build_efficientnetv2_foundation(num_classes=num_classes, variant=variant)
    elif "vit" in name_lower or "transformer" in name_lower:
        return build_vit_foundation(num_classes=num_classes)
    elif "bit" in name_lower or "resnet" in name_lower:
        return build_bit_foundation(num_classes=num_classes)
    else:
        print(f"[Warning] Unknown model '{model_name}', defaulting to ConvNeXt-Base...")
        return build_convnext_foundation(num_classes=num_classes, variant="base")
