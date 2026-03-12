import argparse
from typing import Optional

import h5py


def _print_attrs(obj, indent: int = 0) -> None:
    """打印对象的属性。"""
    if not obj.attrs:
        return
    prefix = " " * indent
    print(f"{prefix}- attrs:")
    for k, v in obj.attrs.items():
        print(f"{prefix}    {k}: {v}")


def _summarize_dataset(dset: h5py.Dataset) -> str:
    """返回 dataset 的简要信息，避免直接打印全部数据。"""
    shape = dset.shape
    dtype = dset.dtype
    summary = f"shape={shape}, dtype={dtype}"

    # 对于标量或一维短向量，可以预览一小部分数据
    try:
        if dset.size > 0 and dset.size <= 16:
            data = dset[()]
            summary += f", data={data}"
    except Exception:
        # 某些 lazy / 压缩数据可能读取失败，不强求
        pass

    return summary


def _visit(name: str, obj, max_items: Optional[int], indent: int = 0) -> None:
    """递归打印 group/dataset 的层级结构。"""
    prefix = " " * indent

    if isinstance(obj, h5py.Group):
        print(f"{prefix}[Group] {name or '/'}")
        _print_attrs(obj, indent + 2)

        keys = list(obj.keys())
        n = len(keys)
        if max_items is not None and n > max_items:
            show_keys = keys[:max_items]
            tail = f"... ({n - max_items} more items omitted)"
        else:
            show_keys = keys
            tail = None

        for k in show_keys:
            child = obj[k]
            _visit(f"{name}/{k}" if name else k, child, max_items, indent + 2)

        if tail:
            print(" " * (indent + 2) + tail)

    elif isinstance(obj, h5py.Dataset):
        info = _summarize_dataset(obj)
        print(f"{prefix}[Dataset] {name}: {info}")
        _print_attrs(obj, indent + 2)

    else:
        print(f"{prefix}[Unknown] {name} ({type(obj)})")


def inspect_hdf5(path: str, max_items: Optional[int]) -> None:
    """按层级结构打印 hdf5 文件内容。

    - `max_items` 用于限制每个 group 打印的子项数量，避免机器人多步数据过长。
    """
    with h5py.File(path, "r") as f:
        print(f"HDF5 file: {path}")
        _visit("", f, max_items=max_items, indent=0)


def main() -> None:
    path='policy/openvla-oft/processed_data/stack_blocks_two/train/episode_0.hdf5'
    max_items = 30
    inspect_hdf5(path, max_items)


if __name__ == "__main__":
    main()

