def get_args(model_name):
    if model_name == 'vanilla_vit':
        return {"image_size": 32, "patch_size": 16, "num_layers": 2, "num_heads": 4, "hidden_dim": 512, "mlp_dim": 768,
                "dropout": 0.1, "attention_dropout": 0.1, "num_classes": 100, "representation_size": 120}

    else:
        print(f"Model '{model_name}' is not supported.")
