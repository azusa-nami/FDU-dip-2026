from pathlib import Path
import shutil

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "datasets" / "raw_data"
DATA_ROOT = ROOT / "datasets" / "data"
CROP_SIZE = 224


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def center_crop_224(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(f"Image is smaller than {CROP_SIZE}x{CROP_SIZE}: {width}x{height}")
    left = (width - CROP_SIZE) // 2
    top = (height - CROP_SIZE) // 2
    return img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))


def prepare_voc() -> int:
    list_path = ROOT / "VOC2012_224_train_png.txt"
    src_dir = RAW_ROOT / "VOCdevkit" / "VOC2012" / "JPEGImages"
    dst_dir = DATA_ROOT / "train" / "synthetic_voc"

    if not list_path.exists():
        raise FileNotFoundError(f"Missing list file: {list_path}")
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing raw VOC directory: {src_dir}")

    reset_dir(dst_dir)
    names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]

    for index, png_name in enumerate(names, start=1):
        src_path = src_dir / png_name.replace(".png", ".jpg")
        dst_path = dst_dir / png_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing raw VOC image: {src_path}")
        with Image.open(src_path) as img:
            center_crop_224(img.convert("RGB")).save(dst_path)
        if index % 2000 == 0 or index == len(names):
            print(f"[VOC] {index}/{len(names)} processed")

    return len(names)


def prepare_real_train() -> tuple[int, int]:
    src_dir = RAW_ROOT / "real89"
    dst_dir = DATA_ROOT / "train" / "real89"

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing raw Berkeley real dataset: {src_dir}")
    if not (src_dir / "blended").exists() or not (src_dir / "transmission_layer").exists():
        raise FileNotFoundError("Expected 'blended' and 'transmission_layer' in raw_data/real89")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    blended = len([p for p in (dst_dir / "blended").iterdir() if p.is_file()])
    transmission = len([p for p in (dst_dir / "transmission_layer").iterdir() if p.is_file()])
    return blended, transmission


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    voc_count = prepare_voc()
    real_blended, real_transmission = prepare_real_train()

    print("\nDone.")
    print(f"train/synthetic_voc: {voc_count}")
    print(f"train/real89/blended: {real_blended}")
    print(f"train/real89/transmission_layer: {real_transmission}")


if __name__ == "__main__":
    main()
