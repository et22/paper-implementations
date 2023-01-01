import torch.nn as nn
from torch import Tensor, linalg, sum, square, mean, abs, bmm
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    # referenced https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
    def __init__(self, style_weight, content_weight, tv_weight):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = vgg16(weights = VGG16_Weights.IMAGENET1K_FEATURES).features
        self.style_layers = [3, 8, 15, 22]
        self.content_layers = [15]
        self.tv_weight = tv_weight
        self.content_weight = content_weight
        self.style_weight = style_weight
    
    def forward(self, output, style_target, content_target):
        loss = self.style_weight * self.style_loss(output, style_target) \
                 + self.content_weight * self.content_loss(output, content_target) \
                 + self.tv_weight * self.tv_loss(output)
        return loss

    def content_loss(self, output, target):
        output_features = self.extract_features(output, self.content_layers)
        target_features = self.extract_features(target, self.content_layers)

        loss = 0
        for out_feat, targ_feat in zip(output_features, target_features):
            _, c, h, w = out_feat.size()
            loss += 1/(c*h*w) * sum(square(out_feat-targ_feat))

        return loss

    def style_loss(self, output, target):
        output_features = self.extract_features(output, self.style_layers)
        target_features = self.extract_features(target, self.style_layers)
        loss = 0
        for out_feat, targ_feat in zip(output_features, target_features):
            g_out = PerceptualLoss.gram(out_feat)
            g_tar = PerceptualLoss.gram(targ_feat)
            loss += sum(square(linalg.matrix_norm(g_out-g_tar)))

        return loss

    def tv_loss(self, output):
        return mean(abs(output[:, :, :, :-1] - output[:, :, :, 1:])) \
                + mean(abs(output[:, :, :-1, :] - output[:, :, 1:, :]))
    
    def extract_features(self, x, layers):
        features = list()
        for index, layer in enumerate(self.vgg16):
            x = layer(x)
            if index in layers:
                features.append(x)
        return features

    @staticmethod
    def gram(x):
        b, c, h, w = x.size()
        g = bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
        return g.div(c*h*w)
