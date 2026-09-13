"""Модель сегментации области манипуляции.

Ключевая идея (как в CAT-Net / PSCC-Net): следы вставки, инпейнта или
повторного сжатия почти не видны в обычном RGB, но хорошо заметны в
высокочастотных доменах. Поэтому на вход энкодеру подаём не только RGB,
но и два дополнительных канала:

  - шумовой остаток (обучаемый высокочастотный фильтр, фильтр Байара);
  - DCT-остаток (высокочастотные коэффициенты 8x8 DCT, ловят JPEG-артефакты).

Итого 5 входных каналов -> энкодер (ResNet) -> U-Net декодер -> маска.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from . import config


class NoiseResidual(nn.Module):
    """Одноканальная карта шума из RGB.

    Перевод в яркость фиксированной линейной комбинацией, затем обучаемый
    высокочастотный фильтр с нулевой суммой весов. Отбрасывает плавные области
    и оставляет шум и резкие границы.
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


class DCTResidual(nn.Module):
    """Высокочастотный DCT-остаток (JPEG-артефакты).

    Раскладываем изображение по блокам 8x8 в базисе DCT-II, зануляем низкие
    частоты и восстанавливаем обратно. Остаётся только высокочастотная
    составляющая, в которой лучше всего видны границы вставки и следы
    повторного сжатия (разные JPEG-сетки у оригинального и вставленного куска).
    """

    def __init__(self, block_size: int = 8, keep_from: int = 16):
        super().__init__()
        self.block_size = block_size
        self.keep_from = keep_from

        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.to_gray.weight.copy_(
                torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            )
        self.to_gray.weight.requires_grad_(False)

        # Базис DCT-II: block_size^2 фильтров размера block_size x block_size.
        self.register_buffer("basis", self._build_dct_basis(block_size))

    @staticmethod
    def _build_dct_basis(n: int) -> torch.Tensor:
        basis = torch.zeros(n * n, 1, n, n)
        for u in range(n):
            for v in range(n):
                cu = 1.0 / math.sqrt(n) if u == 0 else math.sqrt(2.0 / n)
                cv = 1.0 / math.sqrt(n) if v == 0 else math.sqrt(2.0 / n)
                for x in range(n):
                    for y in range(n):
                        basis[u * n + v, 0, x, y] = (
                            cu * cv
                            * math.cos((2 * x + 1) * u * math.pi / (2 * n))
                            * math.cos((2 * y + 1) * v * math.pi / (2 * n))
                        )
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = self.to_gray(x)
        coeff = F.conv2d(gray, self.basis, stride=self.block_size)  # (B, 64, H/8, W/8)
        coeff = coeff.clone()
        coeff[:, : self.keep_from] = 0.0  # зануляем низкочастотные коэффициенты
        # Обратное преобразование (транспонированная свёртка) до исходного размера.
        rec = F.conv_transpose2d(coeff, self.basis, stride=self.block_size)
        return rec


def _patch_encoder_input(encoder: nn.Module, new_in_channels: int) -> None:
    """Расширяет первую свёртку энкодера на дополнительные входные каналы.

    Веса существующих RGB-каналов сохраняются, новые каналы инициализируются
    нулём, чтобы модель на старте вела себя как обычный RGB U-Net.
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
    """U-Net, принимающий одно RGB-изображение и предсказывающий маску правки.

    Внутри дополняет RGB шумовым и DCT-остатком (5 каналов) и пропускает через
    сегментационный U-Net с заданным энкодером.
    """

    def __init__(self, encoder_name: str = "resnet50", pretrained: str = "imagenet",
                 use_dct: bool = True):
        super().__init__()
        self.encoder_name = encoder_name
        self.use_dct = use_dct

        self.noise = NoiseResidual()
        self.dct = DCTResidual() if use_dct else None

        extra_channels = 1 + (1 if use_dct else 0)
        in_channels = 3 + extra_channels

        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=3,
            classes=1,
            activation=None,
        )
        _patch_encoder_input(self.backbone.encoder, new_in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.noise(x)
        if self.dct is not None:
            dct = self.dct(x)
            x_multi = torch.cat([x, residual, dct], dim=1)
        else:
            x_multi = torch.cat([x, residual], dim=1)
        return self.backbone(x_multi)


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
