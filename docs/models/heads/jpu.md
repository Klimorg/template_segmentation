# FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

!!! note "Abstract"

    Modern approaches for semantic segmentation usually employ dilated convolutions in the backbone to extract high-resolution feature maps, which brings heavy computation complexity and memory footprint.

    To replace the time and memory consuming dilated convolutions, we propose a novel joint upsampling module named Joint Pyramid Upsampling (JPU) by formulating the task of extracting high-resolution feature maps into a joint upsampling problem. With the proposed JPU, our method reduces the computation complexity by more than three times without performance loss. Experiments show that JPU is superior to other upsampling modules, which can be plugged into many existing approaches to reduce computation complexity and improve performance.

    By replacing dilated convolutions with the proposed JPU module, our method achieves the state-of-the-art performance in Pascal Context dataset (mIoU of 53.13%) and ADE20K dataset (final score of 0.5584) while running 3 times faster.

    [ArXiv link](https://arxiv.org/abs/1903.11816v1)


!!! note "Architecture"
    ![screenshot](./images/jpu.svg)

::: src.model.jpu
    rendering:
        show_source: true
