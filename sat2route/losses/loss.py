import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, lambda_recon=200.0):
        super(Loss, self).__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.lambda_recon = lambda_recon

    def forward(self, model, real=None, condition=None, mode='generator', gen=None, disc=None):
        if mode == 'generator':
            generator = model if gen is None else gen
            discriminator = disc
            return self.generator_loss(generator, discriminator, real, condition)
        else:
            discriminator = model if disc is None else disc
            generator = gen
            return self.discriminator_loss(generator, discriminator, real, condition)
    
    def generator_loss(self, gen, disc, real, condition):
        gen_logits = gen(condition)
        fake = torch.sigmoid(gen_logits) 

        fake_logits = disc(fake, condition)

        adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
        recon_loss = self.recon_criterion(fake, real)
        total_loss = adv_loss + self.lambda_recon * recon_loss

        return {
            'total': total_loss,
            'adv': adv_loss,
            'recon': recon_loss,
        }
    
    def discriminator_loss(self, gen, disc, real, condition):
        with torch.no_grad():
            fake = gen(condition)
        real_logits = disc(real, condition)
        fake_logits = disc(fake.detach(), condition)

        real_loss = self.adv_criterion(real_logits, torch.ones_like(real_logits))
        fake_loss = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))

        d_loss = (real_loss + fake_loss) / 2

        return {
            'total': d_loss,
            'real': real_loss,
            'fake': fake_loss
        }