# CutMix pipeline

[CutMix](https://arxiv.org/abs/1905.04899v2) is an image data augmentation strategy.

Instead of simply removing pixels as in [Cutout](https://paperswithcode.com/method/cutout), we replace the removed regions with a patch from another image. The ground truth masks are also mixed proportionally.

The added patches further enhance localization ability by requiring the model to identify the object from a partial view.

::: src.pipelines.cutmix
    rendering:
        sort_members: source
        show_source: true
