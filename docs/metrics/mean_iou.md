# Mean Intersection over Union

We have to modify the definition of the Mean IoU if we want to consider sparse outputs for the segmentation. From [StackOverflow](https://stackoverflow.com/questions/61824470/dimensions-mismatch-error-when-using-tf-metrics-meaniou-with-sparsecategorical).

::: src.metrics.mean_iou
    rendering:
        show_source: true
