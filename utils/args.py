def get_args(model_name):

    # CIFAR-10 and CIFAR-100
    try:
        args = {
            'vit_tiny': {"image_size": 32, "patch_size": 4, "num_layers": 7, "num_heads": 4, "hidden_dim": 256,
                     "mlp_dim": 512, "dropout": 0.1, "attention_dropout": 0.1},
            'vit_small': {"image_size": 32, "patch_size": 4, "num_layers": 7, "num_heads": 4, "hidden_dim": 256,
                      "mlp_dim": 512, "dropout": 0.1, "attention_dropout": 0.1},
            'vit_base': {"image_size": 32, "patch_size": 4, "num_layers": 7, "num_heads": 4, "hidden_dim": 256,
                     "mlp_dim": 512, "dropout": 0.1, "attention_dropout": 0.1},
            'vit_large': {"image_size": 32, "patch_size": 4, "num_layers": 7, "num_heads": 4, "hidden_dim": 256,
                      "mlp_dim": 512, "dropout": 0.1, "attention_dropout": 0.1},
            'vit_huge': {"image_size": 32, "patch_size": 4, "num_layers": 7, "num_heads": 4, "hidden_dim": 256,
                     "mlp_dim": 512, "dropout": 0.1, "attention_dropout": 0.1},

            'swin_tiny': {"image_size": 32, "patch_size": [2, 2], "embed_dim": 96, "depths": [2, 2, 6, 2],
                     "num_heads": [3, 6, 12, 24],
                     "window_size": [4, 4], "mlp_ratio": 4.0, "dropout": 0.0, "attention_dropout": 0.0,
                     "stochastic_depth_prob": 0.2},
            'swin_small': {"image_size": 32, "patch_size": [2, 2], "embed_dim": 96, "depths": [2, 2, 18, 2],
                      "num_heads": [3, 6, 12, 24],
                      "window_size": [4, 4], "mlp_ratio": 4.0, "dropout": 0.0, "attention_dropout": 0.0,
                      "stochastic_depth_prob": 0.3},
            'swin_base': {"image_size": 32, "patch_size": [2, 2], "embed_dim": 128, "depths": [2, 2, 18, 2],
                     "num_heads": [4, 8, 16, 32],
                     "window_size": [4, 4], "mlp_ratio": 4.0, "dropout": 0.0, "attention_dropout": 0.0,
                     "stochastic_depth_prob": 0.5},
            'swin_tinv2': {"image_size": 32, "patch_size": [2, 2], "embed_dim": 96, "depths": [2, 2, 6, 2],
                       "num_heads": [3, 6, 12, 24],
                       "window_size": [4, 4], "mlp_ratio": 4.0, "dropout": 0.0, "attention_dropout": 0.0,
                       "stochastic_depth_prob": 0.2},
            'swin_smallv2': {"image_size": 32, "patch_size": [2, 2], "embed_dim": 96, "depths": [2, 2, 18, 2],
                        "num_heads": [3, 6, 12, 24],
                        "window_size": [4, 4], "mlp_ratio": 4.0, "dropout": 0.0, "attention_dropout": 0.0,
                        "stochastic_depth_prob": 0.3},
            'swin_basev2': {"image_size": 32, "patch_size": [2, 2], "embed_dim": 128, "depths": [2, 2, 18, 2],
                       "num_heads": [4, 8, 16, 32],
                       "window_size": [4, 4], "mlp_ratio": 4.0, "dropout": 0.0, "attention_dropout": 0.0,
                       "stochastic_depth_prob": 0.5}
        }
    except:
        print(f"Unknown model name: {model_name}")

    model = '_'.join(model_name.split('_')[:-1])
    final_args = args[model]

    dataset_name = model_name.split("_")[-1]
    if dataset_name.lower() == 'cifar100':
        final_args['num_classes'] = 100
    elif dataset_name.lower() == 'cifar10':
        final_args['num_classes'] = 10
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return final_args


# Example usage:
# args = get_args('swin_tiny_cifar100')
# print(args)

# args = get_args('vanilla_vit_tiny_cifar10')
# print(args)
