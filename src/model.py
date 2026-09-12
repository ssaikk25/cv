"""Модель: U-Net с дополнительным шумовым каналом.

Классический приём в задачах детекции манипуляций: следы инпейнта, вставки
или повторного сжатия часто почти не видны в обычном RGB, но хорошо заметны
в высокочастотной составляющей (карте шума). Поэтому мы выделяем шумовой
остаток обучаемой свёрткой и подаём его энкодеру четвёртым каналом.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from . import config


class NoiseResidual(nn.Module):
    """Выделяет из RGB одноканальную карту шума.

    Сначала изображение переводится в яркость фиксированной линейной
    комбинацией каналов, затем через обучаемый высокочастотный фильтр.
    Фильтр инициализируется так, что сумма его весов равна нулю: он
    отбрасывает плавные области и оставляет шум и резкие границы.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            self.to_gray.weight.copy_(gray_weights)
        self.to_gray.weight.requires_grad_(False)

        self.highpass = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        with torch.no_grad():
            w = torch.full(
                (1, 1, kernel_size, kernel_size),
                1.0 / (kernel_size * kernel_size - 1),
            )
            w[:, :, kernel_size // 2, kernel_size // 2] = -1.0
            self.highpass.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = self.to_gray(x)
        return self.highpass(gray)


def _patch_encoder_input(encoder: nn.Module, new_in_channels: int) -> None:
    """Расширяет первую свёртку энкодера на дополнительный входной канал.

    Веса существующих RGB-каналов сохраняются, а новый канал инициализируется
    нулём, чтобы модель на старте вела себя как обычный RGB U-Net и постепенно
    училась использовать шумовой признак.
    """
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            new_conv = nn.Conv2d(
                new_in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3].copy_(module.weight)
                new_conv.weight[:, 3:].zero_()
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)

            parts = name.split(".")
            parent = encoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            return

    raise RuntimeError("В энкодере не найдена свёртка для расширения входных каналов")


class ManipulationUnet(nn.Module):
    """U-Net, принимающий одно RGB-изображение и предсказывающий маску правки."""

    def __init__(self, encoder_name: str = "resnet34", pretrained: str = "imagenet"):
        super().__init__()
        self.encoder_name = encoder_name
        self.noise = NoiseResidual()
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=3,
            classes=1,
            activation=None,
        )
        _patch_encoder_input(self.backbone.encoder, new_in_channels=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.noise(x)
        x_with_noise = torch.cat([x, residual], dim=1)
        return self.backbone(x_with_noise)


def count_flops(model: nn.Module, input_shape, device: str = "cpu") -> float:
    """Считает строгие FLOPs для одного изображения через FlopCounterMode.

    Возвращает число в гигафлопсах. Именно такой подсчёт эталонный по правилам
    конкурса (в отличие от fvcore/thop, которые на деле считают MAC).
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, *input_shape, device=device)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return float(counter.get_total_flops()) / 1e9


def check_flops(model: nn.Module, img_size: int, device: str = "cpu") -> None:
    """Проверяет, что модель укладывается в лимит 100 GFLOPs на изображение."""
    flops = count_flops(model, (3, img_size, img_size), device)
    print(f"FLOPs при входе {img_size}x{img_size}: {flops:.2f} GFLOPs (лимит {config.MAX_GFLOPS})")
    if flops > config.MAX_GFLOPS:
        raise SystemExit(
            f"Модель превышает лимит {config.MAX_GFLOPS} GFLOPs: {flops:.2f}. "
            "Уменьшите img_size или выберите более лёгкий энкодер."
        )
