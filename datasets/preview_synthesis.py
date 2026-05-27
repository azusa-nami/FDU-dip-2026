from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.reflect_dataset import paired_data_transforms
from data.transforms import ReflectionSythesis_1, AdvancedReflectionSythesis


def parse_args():
    parser = ArgumentParser(description="Preview ERRNet online reflection synthesis.")
    parser.add_argument("--input_dir", default="./datasets/data/train/synthetic_voc")
    parser.add_argument("--output_dir", default="./results/synthesis_preview")
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--legacy", action="store_true", help="use the original ERRNet synthesis")
    parser.add_argument("--low_sigma", type=float, default=2)
    parser.add_argument("--high_sigma", type=float, default=5)
    parser.add_argument("--low_gamma", type=float, default=1.3)
    parser.add_argument("--high_gamma", type=float, default=1.3)
    parser.add_argument("--no_transforms", action="store_true", help="disable the training random resize/crop/flip")
    return parser.parse_args()


def to_uint8(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)


def save_sample(output_dir, index, blended, transmission, reflection, mask=None, glare=None, params=None):
    sample_dir = output_dir / f"{index:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(blended)).save(sample_dir / "input_blended.png")
    Image.fromarray(to_uint8(transmission)).save(sample_dir / "target_transmission.png")
    Image.fromarray(to_uint8(reflection)).save(sample_dir / "reflection_layer.png")
    if mask is not None:
        Image.fromarray(to_uint8(mask)).save(sample_dir / "mask.png")
    if glare is not None:
        Image.fromarray(to_uint8(glare)).save(sample_dir / "glare.png")
    if params is not None:
        (sample_dir / "params.txt").write_text(json.dumps(params, indent=2, sort_keys=True))


def labeled_tile(image, label):
    image = Image.fromarray(to_uint8(image))
    label_h = 24
    tile = Image.new("RGB", (image.width, image.height + label_h), "white")
    tile.paste(image, (0, label_h))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return tile


def make_contact_sheet(rows, output_path):
    if not rows:
        return

    row_images = []
    for row_item in rows:
        blended, transmission, reflection = row_item[:3]
        tiles = [
            labeled_tile(blended, "input"),
            labeled_tile(transmission, "target: B"),
            labeled_tile(reflection, "reflection layer"),
        ]
        if len(row_item) > 3 and row_item[3] is not None:
            tiles.append(labeled_tile(row_item[3], "mask"))
        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        row_images.append(row)

    sheet = Image.new("RGB", (max(row.width for row in row_images), sum(row.height for row in row_images)), "white")
    y = 0
    for row in row_images:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_path)


def pil_to_float(image):
    return np.asarray(image, np.float32) / 255.0


def normalize01(image):
    image = np.asarray(image, np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(image)
    return (image - min_value) / (max_value - min_value)


def apply_color_and_exposure(reflection):
    temperature = np.random.uniform(-0.25, 0.25)
    exposure = np.random.uniform(0.65, 1.35)
    gamma = np.random.uniform(0.75, 1.25)

    gains = np.array([1.0 + temperature, 1.0, 1.0 - temperature], dtype=np.float32)
    reflection = np.clip(reflection * gains.reshape(1, 1, 3), 0, 1)
    reflection = np.clip(reflection * exposure, 0, 1)
    reflection = np.power(reflection, gamma)
    return reflection, {
        "temperature": float(temperature),
        "exposure": float(exposure),
        "reflection_gamma": float(gamma),
    }


def warp_reflection(reflection):
    height, width = reflection.shape[:2]

    angle = np.random.uniform(-18, 18)
    scale = np.random.uniform(0.82, 1.2)
    tx = np.random.uniform(-0.14, 0.14) * width
    ty = np.random.uniform(-0.14, 0.14) * height
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    affine[:, 2] += [tx, ty]
    warped = cv2.warpAffine(
        reflection,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    max_shift = np.random.uniform(0.06, 0.22)
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    jitter = np.float32(
        [
            [np.random.uniform(0, max_shift) * width, np.random.uniform(0, max_shift) * height],
            [width - 1 - np.random.uniform(0, max_shift) * width, np.random.uniform(0, max_shift) * height],
            [width - 1 - np.random.uniform(0, max_shift) * width, height - 1 - np.random.uniform(0, max_shift) * height],
            [np.random.uniform(0, max_shift) * width, height - 1 - np.random.uniform(0, max_shift) * height],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, jitter)
    warped = cv2.warpPerspective(
        warped,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return warped, {
        "affine_angle": float(angle),
        "affine_scale": float(scale),
        "affine_tx": float(tx),
        "affine_ty": float(ty),
        "perspective_max_shift": float(max_shift),
    }


def gaussian_blur(reflection):
    kernel = int(np.random.choice([3, 5, 7, 9, 11, 15]))
    sigma = np.random.uniform(0.35, 3.2)
    return cv2.GaussianBlur(reflection, (kernel, kernel), sigma), {
        "blur": "gaussian",
        "blur_kernel": kernel,
        "blur_sigma": float(sigma),
    }


def motion_blur(reflection):
    kernel_size = int(np.random.choice([5, 7, 9, 11, 15]))
    angle = np.random.uniform(0, 180)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel = kernel / max(kernel.sum(), 1e-6)
    return cv2.filter2D(reflection, -1, kernel), {
        "blur": "motion",
        "blur_kernel": kernel_size,
        "motion_angle": float(angle),
    }


def defocus_blur(reflection):
    radius = int(np.random.choice([1, 2, 3, 4, 5]))
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel = kernel / max(kernel.sum(), 1e-6)
    return cv2.filter2D(reflection, -1, kernel), {
        "blur": "defocus",
        "defocus_radius": radius,
    }


def blur_reflection(reflection):
    if random.random() < 0.3:
        return reflection, {
            "blur": "sharp",
            "blur_applied": False,
            "sharp_mix": 1.0,
        }

    blur_fn = random.choice([gaussian_blur, motion_blur, defocus_blur])
    blurred, params = blur_fn(reflection)
    if random.random() < 0.8:
        sharp_mix = np.random.uniform(0.35, 0.85)
        mixed = np.clip(sharp_mix * reflection + (1.0 - sharp_mix) * blurred, 0, 1)
    else:
        sharp_mix = 0.0
        mixed = blurred

    params.update({
        "blur_applied": True,
        "sharp_mix": float(sharp_mix),
    })
    return mixed, params


def shift_image(image, dx, dy):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def add_ghosting(reflection):
    if random.random() >= 0.85:
        return reflection, {
            "ghost_enabled": False,
            "ghost_beta": 0.0,
            "ghost_shift": [0, 0],
        }

    beta = 0.3
    max_shift = max(6, int(min(reflection.shape[:2]) * 0.12))
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    if dx == 0 and dy == 0:
        dx = max_shift
    ghost = np.clip(reflection + beta * shift_image(reflection, dx, dy), 0, 1)
    return ghost, {
        "ghost_enabled": True,
        "ghost_beta": float(beta),
        "ghost_shift": [int(dx), int(dy)],
    }


def low_frequency_noise(height, width):
    small_h = max(4, height // 32)
    small_w = max(4, width // 32)
    noise = np.random.rand(small_h, small_w).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(width, height) * 0.04)
    return normalize01(noise)


def structured_streak_mask(height, width):
    mask = np.zeros((height, width), dtype=np.float32)
    orientation = random.choice(["vertical", "diagonal"])
    count = np.random.randint(1, 5)

    for _ in range(count):
        strength = np.random.uniform(0.25, 0.8)
        thickness = int(np.random.uniform(0.035, 0.12) * min(height, width))
        if orientation == "vertical":
            x0 = int(np.random.uniform(-0.1, 1.1) * width)
            y0 = -height
            x1 = int(x0 + np.random.uniform(-0.15, 0.15) * width)
            y1 = height * 2
        else:
            start_left = random.random() < 0.5
            x0 = -width if start_left else width * 2
            y0 = int(np.random.uniform(-0.2, 1.0) * height)
            x1 = width * 2 if start_left else -width
            y1 = int(y0 + np.random.uniform(-0.6, 0.6) * height)
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), float(strength), thickness, cv2.LINE_AA)

    sigma = np.random.uniform(8.0, 28.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return normalize01(mask), {
        "structured_mask_enabled": True,
        "structured_mask_orientation": orientation,
        "structured_mask_count": int(count),
        "structured_mask_sigma": float(sigma),
    }


def spatial_mask(height, width):
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = np.random.uniform(0.25, 0.75) * width
    cy = np.random.uniform(0.25, 0.75) * height
    dist = np.sqrt(((x - cx) / max(width, 1)) ** 2 + ((y - cy) / max(height, 1)) ** 2)
    radial = normalize01(dist)
    mode = random.choice(["center", "edge"])
    if mode == "center":
        radial = 1.0 - radial

    noise = low_frequency_noise(height, width)
    mix = np.random.uniform(0.35, 0.75)
    mask = normalize01(mix * radial + (1.0 - mix) * noise)
    structured_params = {
        "structured_mask_enabled": False,
        "structured_mask_orientation": "none",
        "structured_mask_count": 0,
    }
    if random.random() < 0.7:
        streak_mask, structured_params = structured_streak_mask(height, width)
        streak_strength = np.random.uniform(0.25, 0.65)
        mask = normalize01((1.0 - streak_strength) * mask + streak_strength * streak_mask)
        structured_params["structured_mask_strength"] = float(streak_strength)

    mask = np.clip(0.35 + 0.9 * mask, 0, 1)
    params = {
        "mask_mode": mode,
        "mask_mix_radial": float(mix),
        "mask_center": [float(cx), float(cy)],
    }
    params.update(structured_params)
    return mask[..., None], params


def add_glare(height, width):
    glare = np.zeros((height, width, 3), dtype=np.float32)
    if random.random() >= 0.85:
        return glare, {
            "glare_enabled": False,
            "glare_soft_spots": 0,
            "glare_ellipse_spots": 0,
            "glare_streaks": 0,
        }

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    soft_spot_count = np.random.randint(1, 4)
    for _ in range(soft_spot_count):
        cx = np.random.uniform(0, width)
        cy = np.random.uniform(0, height)
        sigma = np.random.uniform(0.03, 0.16) * max(height, width)
        strength = np.random.uniform(0.08, 0.3)
        color = np.array(
            [
                np.random.uniform(0.9, 1.0),
                np.random.uniform(0.82, 1.0),
                np.random.uniform(0.65, 1.0),
            ],
            dtype=np.float32,
        )
        spot = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
        glare += strength * spot[..., None] * color.reshape(1, 1, 3)

    ellipse_count = np.random.randint(1, 5)
    for _ in range(ellipse_count):
        overlay = np.zeros_like(glare)
        center = (int(np.random.uniform(0, width)), int(np.random.uniform(0, height)))
        axes = (
            int(np.random.uniform(0.05, 0.22) * width),
            int(np.random.uniform(0.015, 0.08) * height),
        )
        angle = float(np.random.uniform(-35, 35))
        strength = float(np.random.uniform(0.08, 0.32))
        color = (
            strength,
            strength * np.random.uniform(0.92, 1.0),
            strength * np.random.uniform(0.82, 1.0),
        )
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1, cv2.LINE_AA)
        glare += cv2.GaussianBlur(overlay, (0, 0), np.random.uniform(2.0, 8.0))

    streak_count = np.random.randint(2, 6)
    for _ in range(streak_count):
        overlay = np.zeros_like(glare)
        x0 = int(np.random.uniform(-0.2, 1.0) * width)
        y0 = int(np.random.uniform(0, height))
        length = int(np.random.uniform(0.35, 1.2) * width)
        angle = np.random.uniform(-35, 35) * np.pi / 180.0
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        strength = float(np.random.uniform(0.08, 0.32))
        thickness = int(np.random.choice([1, 2, 3, 4]))
        color = (
            strength,
            strength * np.random.uniform(0.92, 1.0),
            strength * np.random.uniform(0.82, 1.0),
        )
        cv2.line(overlay, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        blur_sigma = np.random.uniform(2.0, 10.0)
        glare += cv2.GaussianBlur(overlay, (0, 0), blur_sigma)

    glare = np.clip(glare, 0, 1)
    return glare, {
        "glare_enabled": True,
        "glare_soft_spots": int(soft_spot_count),
        "glare_ellipse_spots": int(ellipse_count),
        "glare_streaks": int(streak_count),
    }


def advanced_synthesis(transmission_pil, reflection_pil):
    background = pil_to_float(transmission_pil)
    reflection = pil_to_float(reflection_pil)

    alpha = np.random.uniform(0.18, 0.65)
    tau = np.random.uniform(0.55, 0.95)

    reflection, color_params = apply_color_and_exposure(reflection)
    reflection, warp_params = warp_reflection(reflection)
    reflection, blur_params = blur_reflection(reflection)
    reflection, ghost_params = add_ghosting(reflection)

    height, width = background.shape[:2]
    mask, mask_params = spatial_mask(height, width)
    glare, glare_params = add_glare(height, width)
    noise_sigma = np.random.uniform(0.003, 0.025)
    noise = np.random.normal(0.0, noise_sigma, background.shape).astype(np.float32)

    reflection_layer = np.clip(mask * reflection, 0, 1)
    blended = tau * background + alpha * reflection_layer + glare + noise
    blended = np.clip(blended, 0, 1)

    params = {
        "alpha": float(alpha),
        "tau": float(tau),
        "noise_sigma": float(noise_sigma),
    }
    params.update(color_params)
    params.update(warp_params)
    params.update(blur_params)
    params.update(ghost_params)
    params.update(mask_params)
    params.update(glare_params)
    return background, reflection_layer, blended, mask, glare, params


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two images in {input_dir}")

    midpoint = len(paths) // 2
    b_paths = paths[:midpoint]
    r_paths = paths[midpoint : midpoint * 2]

    synth = None
    if args.legacy:
        synth = ReflectionSythesis_1(
            kernel_sizes=[11],
            low_sigma=args.low_sigma,
            high_sigma=args.high_sigma,
            low_gamma=args.low_gamma,
            high_gamma=args.high_gamma,
        )
    else:
        synth = AdvancedReflectionSythesis()

    rows = []
    for index in range(args.num):
        b_path = b_paths[index % len(b_paths)]
        r_path = r_paths[index % len(r_paths)]
        transmission = Image.open(b_path).convert("RGB")
        reflection = Image.open(r_path).convert("RGB")

        if not args.no_transforms:
            transmission, reflection = paired_data_transforms(transmission, reflection)

        if args.legacy:
            transmission, reflection, blended = synth(transmission, reflection)
            mask = None
            glare = None
            params = {"mode": "legacy"}
        else:
            transmission, reflection, blended, mask, glare, params = synth(transmission, reflection, return_extras=True)

        save_sample(output_dir, index, blended, transmission, reflection, mask=mask, glare=glare, params=params)
        rows.append((blended, transmission, reflection, mask))

    make_contact_sheet(rows, output_dir / "preview_grid.png")
    print(f"saved {args.num} samples to {output_dir}")
    print(f"contact sheet: {output_dir / 'preview_grid.png'}")


if __name__ == "__main__":
    main()

"""
uv run  python datasets/preview_synthesis.py --num 12 --output_dir ./results/synthesis_preview_advanced
"""
