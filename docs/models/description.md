# Architecture description and backbone/segmentations heads compatibility

Modern, State-of-the-art semantic segmentation models rely on the combination of multiple (most of the time 2) networks :

* The network responsible of features extractions, usually a CNN like ResNet. This network is known as the *backbone* of the segmentation model.
* The networks responsible of computing the mask, given the outputs of the backbone. This network is known as the *segmentation head* of the segmentation model.

Compared to the case where a backbone network like ResNet is used for a classification task and only outputs probabilities. Backbone networks in segmentation task are setup to ouptuts one of more feature maps.

!!! info "Definition"

    The ratio between the height/width of the input image and the heights/widths of the outputs feature maps is called the **output stride**. It is usually denoted by OS.

    \[
        \mathrm{OS} := \frac{\text{height-width input}}{\text{height-width output}}
    \]

The segmentation head available in this project are the following ones. For example, OS 4-8-16-32 here means that the segmentation head takes as inputs 4 feature maps coming from the backbone, and denotes the different output strides of the inputs :

* the first input is of output stride 4, meaning that the feature map is 4 times smaller than the orginal input image of the backbone.
* the second input is of output stride 8, meaning that the feature map is 8 times smaller than the orginal input image of the backbone.
* the third input is of output stride 16, meaning that the feature map is 16 times smaller than the orginal input image of the backbone.
* the fourth input is of output stride 32, meaning that the feature map is 32 times smaller than the orginal input image of the backbone.

|  **Segmentation Heads**  |     FPN      |    JPU     | KSAC  | OCNet | ASPP  | ASPP_OCNet |
| :----------------------: | :----------: | :--------: | :---: | :---: | :---: | :--------: |
| **Output stride inputs** | OS 4-8-16-32 | OS 8-16-32 | OS 4  | OS 8  | OS 4  |    OS 4    |
| **Output stride output** |     OS 4     |    OS 4    | OS 4  | OS 8  | OS 4  |    OS 4    |

The backbones available are the following ones.

* GhostNet
* MobileNetv2
* ResNet50
* ResNet50v2
* ResNet101
* ResNet101v2
* VoVNet27
* VoVNet39
* VoVNet57

They are all setup to outputs features maps of outputs strides 4, 8, 16, and 32. The outputs-inputs correspodances are automatically made within the script.
