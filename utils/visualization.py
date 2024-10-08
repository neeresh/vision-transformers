from matplotlib import pyplot as plt
import seaborn as sns


def plot_patches(patched_images, images, labels, num_images=3):
    num_patches = patched_images.shape[1]
    fig, ax = plt.subplots(nrows=num_images, ncols=num_patches + 1, figsize=(15, 5))
    for i in range(num_images):
        for j in range(num_patches):
            ax[i, j].imshow(patched_images[i, j].permute(1, 2, 0).detach().cpu())
            ax[i, j].set_title(f"Patch {j + 1}")
            ax[i, j].axis('off')
        ax[i, num_patches].imshow(images[i].permute(1, 2, 0).detach().cpu())
        ax[i, num_patches].set_title(labels[i].item())
        ax[i, num_patches].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_attention_maps(attention_weights, num_heads):
    # https://github.com/jeonsworld/ViT-pytorch/tree/main -> Correct implementation of attention maps for visualizing.
    for layer_idx, layer_attn_weights in enumerate(attention_weights):
        for head_idx in range(num_heads):
            plt.figure(figsize=(10, 10))
            sns.heatmap(layer_attn_weights[0, head_idx].detach().cpu().numpy(), cmap="viridis")
            plt.title(f"Layer {layer_idx + 1}, Head {head_idx + 1}")
            plt.show()
