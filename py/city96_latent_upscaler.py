# https://github.com/city96/SD-Latent-Upscaler
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

class Upscaler(nn.Module):
	"""
		Basic NN layout, ported from:
		https://github.com/city96/SD-Latent-Upscaler/blob/main/upscaler.py
	"""
	version = 2.1 # network revision
	def head(self):
		return [
			nn.Conv2d(self.chan, self.size, kernel_size=self.krn, padding=self.pad),
			nn.ReLU(),
			nn.Upsample(scale_factor=self.fac, mode="nearest"),
			nn.ReLU(),
		]
	def core(self):
		layers = []
		for _ in range(self.depth):
			layers += [
				nn.Conv2d(self.size, self.size, kernel_size=self.krn, padding=self.pad),
				nn.ReLU(),
			]
		return layers
	def tail(self):
		return [
			nn.Conv2d(self.size, self.chan, kernel_size=self.krn, padding=self.pad),
		]

	def __init__(self, fac, depth=16):
		super().__init__()
		self.size = 64      # Conv2d size
		self.chan = 4       # in/out channels
		self.depth = depth  # no. of layers
		self.fac = fac      # scale factor
		self.krn = 3        # kernel size
		self.pad = 1        # padding

		self.sequential = nn.Sequential(
			*self.head(),
			*self.core(),
			*self.tail(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.sequential(x)


class LatentUpscaler:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"samples": ("LATENT", ),
				"latent_ver": (["v1", "xl"],),
				"scale_factor": (["1.25", "1.5", "2.0"],),
			}
		}

	RETURN_TYPES = ("LATENT",)
	FUNCTION = "upscale"
	CATEGORY = "latent"

	def upscale(self, samples, latent_ver, scale_factor):
		model = Upscaler(scale_factor)
		weights = str(hf_hub_download(
			repo_id="city96/SD-Latent-Upscaler",
			filename=f"latent-upscaler-v{model.version}_SD{latent_ver}-x{scale_factor}.safetensors")
		)
		# weights = f"./latent-upscaler-v{model.version}_SD{latent_ver}-x{scale_factor}.safetensors"

		model.load_state_dict(load_file(weights))
		lt = samples["samples"]
		lt = model(lt)
		del model
		return ({"samples": lt},)