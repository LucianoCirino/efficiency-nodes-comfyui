# https://github.com/chrisgoringe/cg-noise
import torch

def get_mixed_noise_function(original_noise_function, variation_seed, variation_weight):
    def prepare_mixed_noise(latent_image:torch.Tensor, seed, batch_inds):
        single_image_latent = latent_image[0].unsqueeze_(0)
        different_noise = original_noise_function(single_image_latent, variation_seed, batch_inds)
        original_noise = original_noise_function(single_image_latent, seed, batch_inds)
        if latent_image.shape[0]==1:
            mixed_noise = original_noise * (1.0-variation_weight) + different_noise * (variation_weight)
        else:
            mixed_noise = torch.empty_like(latent_image)
            for i in range(latent_image.shape[0]):
                mixed_noise[i] = original_noise * (1.0-variation_weight*i) + different_noise * (variation_weight*i)
        return mixed_noise
    return prepare_mixed_noise
