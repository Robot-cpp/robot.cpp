#!/usr/bin/env bash

readonly STARVLA_REVISION=631aae02afe6d95876e923ff518e8ff2ab9a2f88

_set_starvla_variant() {
    MODEL_TYPE=$1
    FRAMEWORK=$2
    CHECKPOINT_REVISION=$3
    CHECKPOINT_SIZE=$4
    CHECKPOINT_SHA256=$5
    CHECKPOINT_DIRECTORY=$6
    CHECKPOINT_RELATIVE_PATH=$7
    QWEN_REVISION=$8
    QWEN_DIRECTORY=$9
    DEFAULT_UNNORM_KEY=${10}
    ARTIFACT_STEM=${11}
    REFERENCE_SERVER_NAME=${12}
}

load_starvla_variant() {
    case "$1" in
        oft) _set_starvla_variant starvla oft c3fc8f028429ba14819bf3b16e098776b670c889 9785060316 371cb744227687bb99bcad7f9ff2250cf06da75631359ad3eba4c6bc52570607 \
            oft-bridge-rt1 checkpoints/steps_5000_pytorch_model.pt ebb281ec70b05090aa6165b016eac8ec08e71b17 qwen3-vl-4b-instruct oxe_bridge oft serve_starvla_oft_reference.py ;;
        groot) _set_starvla_variant starvla groot 12acc0b0f1f6230df21c479934a67a930b52f878 9976845210 769d6c400d582a86ae8df8b0b445240ab679dbe77eeb72a4db71e43cd129c7c3 \
            groot-bridge-rt1 checkpoints/steps_20000_pytorch_model.pt ebb281ec70b05090aa6165b016eac8ec08e71b17 qwen3-vl-4b-instruct oxe_bridge groot serve_starvla_groot_reference.py ;;
        pi_v3) _set_starvla_variant starvla pi_v3 99a3c01b3977e6442871a1fb62ce178279c5c3ed 10922634912 7f59a5d0fa9c167fabd941bca8e606bdf5597bfb4f99ca83e345672dd9c345ed \
            pi-v3-bridge-rt1 checkpoints/steps_50000_pytorch_model.pt ebb281ec70b05090aa6165b016eac8ec08e71b17 qwen3-vl-4b-instruct oxe_bridge pi-v3 serve_starvla_bridge_reference.py ;;
        qwen25_oft) _set_starvla_variant starvla oft 11fa6440835ba3e912de43cfe8521043360ffc02 8215912766 51fe8d22c8d57116c2f59c5fdb24323fa3411149e888b807edba99b8354e0861 \
            qwen25-oft-bridge-rt1 checkpoints/steps_10000_pytorch_model.pt 66285546d2b821cf421d4f5eb2576359d3770cd3 qwen2.5-vl-3b-instruct bridge_dataset qwen25-oft serve_starvla_oft_reference.py ;;
        qwen25_groot) _set_starvla_variant starvla groot 5ebc661ba38b29c28f20fff6574801e6f49f3466 8456891339 9646da2ae0b32589a75c8cc88fae96c93c5d269b69fd7a29200744936e01d96f \
            qwen25-groot-bridge-rt1 checkpoints/steps_30000_pytorch_model.pt ce86bd9a53416527b8361e8dfc47316288ffa110 qwen2.5-vl-3b-instruct-action oxe_bridge qwen25-groot serve_starvla_bridge_reference.py ;;
        qwen25_pi) _set_starvla_variant starvla pi 26d0e079fbe3bc3fc62301f44f0025ef7c64ee22 10103104403 8a0e47858921924d5038f7c4393dee6682b83175a85546e35e357e8f74ce8343 \
            qwen25-pi-bridge-rt1 checkpoints/steps_30000_pytorch_model.pt ce86bd9a53416527b8361e8dfc47316288ffa110 qwen2.5-vl-3b-instruct-action oxe_bridge qwen25-pi serve_starvla_bridge_reference.py ;;
        qwen25_fast) _set_starvla_variant starvla fast d9e2977d21755e78a0dd5f9a61586075a636d669 8146439050 f30e89a6b2a166fa3f48af42d5cffde07be44074b861abc7b57e1ccdb734e81e \
            qwen25-fast-bridge-rt1 checkpoints/steps_10000_pytorch_model.pt ce86bd9a53416527b8361e8dfc47316288ffa110 qwen2.5-vl-3b-instruct-action bridge_dataset qwen25-fast serve_starvla_bridge_reference.py ;;
        *) echo "unsupported VARIANT=$1; expected oft, groot, pi_v3, qwen25_oft, qwen25_groot, qwen25_pi, or qwen25_fast" >&2; return 2 ;;
    esac
}
