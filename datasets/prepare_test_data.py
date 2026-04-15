from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "datasets" / "raw_data"
PROCESSED_ROOT = ROOT / "datasets" / "processed_data"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_files(path: Path) -> int:
    return len([p for p in path.iterdir() if p.is_file()])


def copy_paired_dataset(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    if not (src_dir / "blended").exists() or not (src_dir / "transmission_layer").exists():
        raise FileNotFoundError(f"Expected paired dataset layout in {src_dir}")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    blended = count_files(dst_dir / "blended")
    transmission = count_files(dst_dir / "transmission_layer")
    return blended, transmission


def sir2_gt_name(blended_name: str) -> str:
    if "-m-" in blended_name:
        return blended_name.replace("-m-", "-g-")
    if "-m." in blended_name:
        return blended_name.replace("-m.", "-g.")
    raise RuntimeError(f"Unsupported SIR2 blended filename: {blended_name}")


def prepare_sir2_dataset(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    blended_src = src_dir / "blended"
    transmission_src = src_dir / "transmission_layer"
    blended_dst = dst_dir / "blended"
    transmission_dst = dst_dir / "transmission_layer"

    if not blended_src.exists() or not transmission_src.exists():
        raise FileNotFoundError(f"Expected paired SIR2 dataset layout in {src_dir}")

    reset_dir(dst_dir)
    blended_dst.mkdir(parents=True, exist_ok=True)
    transmission_dst.mkdir(parents=True, exist_ok=True)

    for blended_path in sorted(p for p in blended_src.iterdir() if p.is_file()):
        gt_name = sir2_gt_name(blended_path.name)
        gt_path = transmission_src / gt_name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing SIR2 transmission image: {gt_path}")
        shutil.copy2(blended_path, blended_dst / blended_path.name)
        shutil.copy2(gt_path, transmission_dst / blended_path.name)

    return count_files(blended_dst), count_files(transmission_dst)


def prepare_ceilnet_table2() -> tuple[int, int]:
    src_dir = RAW_ROOT / "CEILNet" / "testdata_reflection_synthetic_table2"
    dst_dir = PROCESSED_ROOT / "testdata_CEILNET_table2"
    blended_dir = dst_dir / "blended"
    transmission_dir = dst_dir / "transmission_layer"

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing CEILNet test set: {src_dir}")

    reset_dir(blended_dir)
    reset_dir(transmission_dir)

    inputs = sorted(src_dir.glob("*-input.png"))
    for input_path in inputs:
        stem = input_path.name.replace("-input.png", "")
        label1_path = src_dir / f"{stem}-label1.png"
        if not label1_path.exists():
            raise FileNotFoundError(f"Missing transmission image: {label1_path}")
        shutil.copy2(input_path, blended_dir / f"{stem}.png")
        shutil.copy2(label1_path, transmission_dir / f"{stem}.png")

    return count_files(blended_dir), count_files(transmission_dir)


def merge_sir2(datasets: list[Path], dst_dir: Path) -> tuple[int, int]:
    blended_dir = dst_dir / "blended"
    transmission_dir = dst_dir / "transmission_layer"
    reset_dir(blended_dir)
    reset_dir(transmission_dir)

    seen = set()
    for src_dir in datasets:
        for name in sorted(p.name for p in (src_dir / "blended").iterdir() if p.is_file()):
            if name in seen:
                raise RuntimeError(f"Duplicate filename while merging SIR2 datasets: {name}")
            seen.add(name)
            shutil.copy2(src_dir / "blended" / name, blended_dir / name)
            shutil.copy2(src_dir / "transmission_layer" / name, transmission_dir / name)

    return count_files(blended_dir), count_files(transmission_dir)


def main() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    ceil_blended, ceil_transmission = prepare_ceilnet_table2()
    real20_blended, real20_transmission = copy_paired_dataset(
        RAW_ROOT / "robustsirr_test_dataset" / "real20",
        PROCESSED_ROOT / "real20",
    )
    postcard_blended, postcard_transmission = prepare_sir2_dataset(
        RAW_ROOT / "robustsirr_test_dataset" / "SIR2" / "PostcardDataset",
        PROCESSED_ROOT / "postcard",
    )
    solid_blended, solid_transmission = prepare_sir2_dataset(
        RAW_ROOT / "robustsirr_test_dataset" / "SIR2" / "SolidObjectDataset",
        PROCESSED_ROOT / "objects",
    )
    wild_blended, wild_transmission = prepare_sir2_dataset(
        RAW_ROOT / "robustsirr_test_dataset" / "SIR2" / "WildSceneDataset",
        PROCESSED_ROOT / "wild",
    )
    sir2_blended, sir2_transmission = merge_sir2(
        [
            PROCESSED_ROOT / "postcard",
            PROCESSED_ROOT / "objects",
            PROCESSED_ROOT / "wild",
        ],
        PROCESSED_ROOT / "sir2_withgt",
    )

    print("Done.")
    print(f"testdata_CEILNET_table2/blended: {ceil_blended}")
    print(f"testdata_CEILNET_table2/transmission_layer: {ceil_transmission}")
    print(f"real20/blended: {real20_blended}")
    print(f"real20/transmission_layer: {real20_transmission}")
    print(f"postcard/blended: {postcard_blended}")
    print(f"postcard/transmission_layer: {postcard_transmission}")
    print(f"objects/blended: {solid_blended}")
    print(f"objects/transmission_layer: {solid_transmission}")
    print(f"wild/blended: {wild_blended}")
    print(f"wild/transmission_layer: {wild_transmission}")
    print(f"sir2_withgt/blended: {sir2_blended}")
    print(f"sir2_withgt/transmission_layer: {sir2_transmission}")


if __name__ == "__main__":
    main()
