from data.mpi3d import MPI3DDataset, get_mpi3d_loaders
from data.clevr import CLEVRDataset, get_clevr_loaders
from data.celeba import CelebADataset, get_celeba_loaders
from data.cub200 import CUB200Dataset, get_cub200_loaders


def get_dataloaders(config):
    """Factory function to get dataloaders based on config."""
    dataset_name = config["dataset"]["name"]

    if dataset_name == "mpi3d":
        return get_mpi3d_loaders(config)
    elif dataset_name == "clevr":
        return get_clevr_loaders(config)
    elif dataset_name == "celeba":
        return get_celeba_loaders(config)
    elif dataset_name == "cub200":
        return get_cub200_loaders(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_num_concepts(config):
    """Get number of concepts for a dataset."""
    dataset_name = config["dataset"]["name"]

    if dataset_name == "mpi3d":
        # MPI3D: 7 factors (object_color, object_shape, object_size,
        #         camera_height, background_color, horizontal_axis, vertical_axis)
        # Each factor has multiple values -> total binary concepts
        return 40  # Total number of binary concept indicators
    elif dataset_name == "clevr":
        # CLEVR: shape(3), color(8), size(2), material(2) = ~15 binary concepts per object
        return 30
    elif dataset_name == "celeba":
        return len(config["dataset"].get("celeba_attrs", []))
    elif dataset_name == "cub200":
        return config["dataset"].get("cub200_num_concepts", 20)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
