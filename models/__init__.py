from models.backbone import SharedBackbone
from models.baseline import BaselineModel
from models.cores import CoResModel
# from models.components import ConceptBranch, ResidualBranch
from models.vcores import VCoResModel, VCoResLoss, VariationalConceptBranch
from models.seqcores import SeqCoResModel, SeqCoResLoss


def build_model(config, num_concepts):
    """Factory function to build model based on config."""
    model_type = config.get("_model_type", "cores")
    latent_dim = config["model"]["latent_dim"]

    if model_type == "baseline":
        return BaselineModel(
            latent_dim=latent_dim,
            num_concepts=num_concepts,
            method=config["model"]["baseline"]["method"],
            temperature=config["model"]["baseline"]["temperature"],
        )
    elif model_type == "cores":
        concept_dim = config["model"]["cores"]["concept_dim"]
        residual_dim = config["model"]["cores"]["residual_dim"]

        return CoResModel(
            latent_dim=latent_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            residual_dim=residual_dim,
            use_soft_concepts=config["model"]["cores"]["use_soft_concepts"],
            concept_temperature=config["model"]["cores"]["concept_temperature"],
            aggregation=config["model"]["cores"]["aggregation"],
        )
    elif model_type == "vcores":
        vcores_cfg = config["model"].get("vcores", config["model"]["cores"])
        concept_dim = vcores_cfg["concept_dim"]
        residual_dim = vcores_cfg["residual_dim"]
        image_size = config["dataset"].get("image_size", 64)

        return VCoResModel(
            latent_dim=latent_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            residual_dim=residual_dim,
            use_soft_concepts=vcores_cfg.get("use_soft_concepts", True),
            concept_temperature=vcores_cfg.get("concept_temperature", 1.0),
            aggregation=vcores_cfg.get("aggregation", "sum"),
            image_size=image_size,
            use_decoder=vcores_cfg.get("use_decoder", True),
        )
    elif model_type == "seqcores":
        sc_cfg = config["model"].get("seqcores", {})
        sc_train = config["training"].get("seqcores", {})
        image_size = config["dataset"].get("image_size", 64)

        return SeqCoResModel(
            latent_dim=latent_dim,
            num_codes=sc_cfg.get("num_codes", 64),
            code_dim=sc_cfg.get("code_dim", 32),
            hidden_dim=sc_cfg.get("hidden_dim", 256),
            residual_dim=sc_cfg.get("residual_dim", 32),
            max_steps=sc_cfg.get("max_steps", 8),
            num_concepts=num_concepts,
            gumbel_tau_init=sc_train.get("gumbel_tau_init", 1.0),
            gumbel_tau_min=sc_train.get("gumbel_tau_min", 0.1),
            commitment_cost=sc_cfg.get("commitment_cost", 0.25),
            image_size=image_size,
            use_decoder=sc_cfg.get("use_decoder", True),
            num_gru_layers=sc_cfg.get("num_gru_layers", 1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
