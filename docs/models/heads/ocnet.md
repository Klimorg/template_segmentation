# OCNet: Object Context for Semantic Segmentation

!!! note "Abstract"

    In this paper, we address the semantic segmentation task with a new context aggregation scheme named object context, which focuses on enhancing the role of object information. Motivated by the fact that the category of each pixel is inherited from the object it belongs to, we define the object context for each pixel as the set of pixels that belong to the same category as the given pixel in the image. We use a binary relation matrix to represent the relationship between all pixels, where the value one indicates the two selected pixels belong to the same category and zero otherwise.

    We propose to use a dense relation matrix to serve as a surrogate for the binary relation matrix. The dense relation matrix is capable to emphasize the contribution of object information as the relation scores tend to be larger on the object pixels than the other pixels. Considering that the dense relation matrix estimation requires quadratic computation overhead and memory consumption w.r.t. the input size, we propose an efficient interlaced sparse self-attention scheme to model the dense relations between any two of all pixels via the combination of two sparse relation matrices.

    To capture richer context information, we further combine our interlaced sparse self-attention scheme with the conventional multi-scale context schemes including pyramid pooling and atrous spatial pyramid pooling. We empirically show the advantages of our approach with competitive performances on five challenging benchmarks including: Cityscapes, ADE20K, LIP, PASCAL-Context and COCO-Stuff

    [ArXiv link](https://arxiv.org/abs/1809.00916)


!!! note "Architecture"
    ![screenshot](./images/ocnet.svg)

::: src.model.layers.object_context
    rendering:
        show_source: true


::: src.model.ocnet
    rendering:
        show_source: true
