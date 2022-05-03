from statistics import mode
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput

class Encoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, device):
        super(Encoder, self).__init__()
        
        self.mean_layers = nn.Sequential(
            nn.Linear(s_img*s_img*3, hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], latent_dims)
        )
        self.sigma_layers = nn.Sequential(
            nn.Linear(s_img*s_img*3, hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], latent_dims),
        )

        #distribution setup
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def kull_leib(self, mu, sigma):
        return (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

    def reparameterize(self, mu, sig):
        return mu + sig*self.N.sample(mu.shape)

    def forward(self, batch:torch.Tensor)-> torch.Tensor:

        x = batch["model_input"]['rgb'].flatten(start_dim=1)

        sig = self.sigma_layers(x)
        sig = torch.exp(sig) #because it has to stay positive
        mu = self.mean_layers(x)
        
        #reparameterize to find z
        z = self.reparameterize(mu, sig)

        #loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, device):
        super(Decoder, self).__init__()
        
        self.s_img=s_img
        self.image_layers = nn.Sequential(
            nn.Linear(latent_dims, hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], s_img*s_img*3),
        )
        self.mask_layers = nn.Sequential(
            nn.Linear(latent_dims, hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], s_img*s_img),
        )

    def forward(self, z:torch.Tensor)-> torch.Tensor:
    
        image = self.image_layers(z)
        image = image.reshape((-1, self.s_img, self.s_img, 3))
        mask = self.mask_layers(z)
        mask = mask.reshape((-1, self.s_img, self.s_img))

        return image, mask

class LitVAE(pl.LightningModule):
    def __init__(self, latent_dims, s_img, hdim, device):
        super().__init__()
        self.encoder = Encoder(latent_dims, s_img, hdim, device)
        self.decoder = Decoder(latent_dims, s_img, hdim, device)

    def forward(self, batch: TrainingSample) -> ModelOutput:
        z = self.encoder(batch)
        image, mask = self.decoder(z)
        model_outputs = ModelOutput(
            rgb_with_object=image,
            soft_object_mask=mask,
        )
        return model_outputs

    def configure_optimizers(self) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: TrainingSample, batch_idx: int) -> nn.Module:
        model_targets = batch["model_target"]
        model_outputs = self(batch=batch)
        image = model_outputs['rgb_with_object']
        mask = model_outputs['soft_object_mask']

        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=mask,
            target=model_targets["object_mask"].float()
        )
        rgb_mse_loss = F.mse_loss(
            input=image,
            target=model_targets["rgb_with_object"],
        )
        loss = mask_cross_entropy_loss + rgb_mse_loss + self.encoder.kl*0.1
        self.log("ce_loss", mask_cross_entropy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("mse_loss", rgb_mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("kl_loss", self.encoder.kl*0.1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # print(f"mask_cross_entropy_loss {mask_cross_entropy_loss}, rgb_mse_loss {rgb_mse_loss}, self.encoder.kl: {self.encoder.kl}")
        return loss


