class ModelError(ValueError):
    """[summary]

    Args:
        ValueError ([type]): [description]
    """


class NumClassesError(ModelError):
    """[summary]

    Args:
        ModelError ([type]): [description]
    """


def validate_num_classes(num_classes_model, num_classes_dataset):

    if num_classes_model == num_classes_dataset:
        pass
    else:
        raise NumClassesError(num_classes_model, num_classes_dataset)
