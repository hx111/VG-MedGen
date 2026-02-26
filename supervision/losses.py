import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def charbonnier_penalty(x, epsilon_squared=0.01):
    charbonnier_loss = torch.sqrt(x * x + epsilon_squared)
    return charbonnier_loss

def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def dice_loss(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    loss = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss


def histogram_loss(image1, image2, bins=64):
    image1 = (image1 + 1) / 2.0
    image2 = (image2 + 1) / 2.0

    loss = 0.0
    for ch in range(image1.size(1)):
        hist1 = torch.histc(image1[:, ch, :, :], bins=bins, min=0, max=1)
        hist2 = torch.histc(image2[:, ch, :, :], bins=bins, min=0, max=1)
        loss += F.l1_loss(torch.log1p(hist1), torch.log1p(hist2))

    return loss / image1.size(1)


def moment_loss(image1, image2):
    loss = 0.0
    for ch in range(image1.size(1)):
        ch_1 = image1[:, ch, :, :]
        ch_2 = image2[:, ch, :, :]

        mean1 = torch.mean(ch_1)
        mean2 = torch.mean(ch_2)
        loss += F.l1_loss(mean1, mean2)

        std1 = torch.std(ch_1)
        std2 = torch.std(ch_2)
        loss += F.l1_loss(std1, std2)

    return loss


def local_color_loss(reco_img, orig_img, mask, loss_fn=F.l1_loss):
    num_channels = orig_img.size(1)

    mask_sum = mask.sum(dim=[2, 3], keepdim=True) + 1e-5

    reco_masked = reco_img * mask
    orig_masked = orig_img * mask

    avg_color_reco = reco_masked.sum(dim=[2, 3])
    avg_color_orig = orig_masked.sum(dim=[2, 3])

    mask_sum_reshaped = mask_sum.view(mask_sum.size(0), -1)

    avg_color_reco = avg_color_reco / mask_sum_reshaped
    avg_color_orig = avg_color_orig / mask_sum_reshaped

    loss = loss_fn(avg_color_reco, avg_color_orig)

    return loss

def dice_score(pred, target):

    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    score = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()
    
    return score


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
        target = target.view(-1,1).long()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()

        self.feature_extractor = nn.Sequential(*list(vgg.children())[:12]).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.loss_fn = nn.L1Loss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, generated_img, target_img):
        gen_norm = (generated_img + 1) / 2.0
        tgt_norm = (target_img + 1) / 2.0

        gen_norm = (gen_norm - self.mean) / self.std
        tgt_norm = (tgt_norm - self.mean) / self.std

        gen_features = self.feature_extractor(gen_norm)
        tgt_features = self.feature_extractor(tgt_norm)

        loss = self.loss_fn(gen_features, tgt_features)

        return loss