"""Dataset manifest: one row per audio file, with its label and its split.

Labels come ONLY from the folder name (normal / abnormal). Splits are assigned here, on whole files,
BEFORE any windowing, so overlapping windows can never leak between train, validation and test.
"""
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"normal": 0, "abnormal": 1}


def natural_key(text: str):
    """Sort 'file2' before 'file10' (and keep zero-padded names in order)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


# ----------------------------------------------------------------------------- scanning
def scan_folder(root, cfg: dict):
    """Find every WAV under root/<type>/<id>/<label>/. Returns (DataFrame, list of unexpected paths)."""
    root = Path(root)
    types = set(cfg["dataset"]["machine_types"])
    ids = set(cfg["dataset"]["machine_ids"])
    rows, unexpected = [], []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        parts = p.relative_to(root).parts
        if p.suffix.lower() != ".wav" or len(parts) != 4 or parts[0] not in types \
                or parts[1] not in ids or parts[2] not in LABELS:
            if p.suffix.lower() != ".csv":           # inventory CSVs are expected
                unexpected.append(p.relative_to(root).as_posix())
            continue
        mtype, mid, label_name, fname = parts
        rows.append({"file_id": f"{mtype}_{mid}_{label_name}_{Path(fname).stem}",
                     "path": p.relative_to(root).as_posix(),
                     "machine_type": mtype, "machine_id": mid, "group": f"{mtype}_{mid}",
                     "label_name": label_name, "label": LABELS[label_name],
                     "snr_db": cfg["dataset"]["snr_db"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, unexpected
    # natural order of the file names inside each machine and label (this order defines the "blocks")
    df["_k"] = df["path"].map(lambda s: tuple(natural_key(Path(s).name)))
    df = df.sort_values(["machine_type", "machine_id", "label_name", "_k"], kind="stable")
    df = df.drop(columns="_k").reset_index(drop=True)
    df["index_in_label"] = df.groupby(["machine_type", "machine_id", "label_name"]).cumcount()
    return df, unexpected


# ----------------------------------------------------------------------------- anomaly-detector splits
def _group_seed(seed: int, group: str) -> int:
    return (seed + zlib.crc32(group.encode("utf-8"))) % (2 ** 32)


def _block_split(normal_pos, abn_pos, an_cfg, group):
    """Return dict split -> list of row positions for ONE machine. Layout for normal files:
    [validation block][gap][ ... training ... ][gap][test block]. Abnormal: [validation][gap][test]."""
    b = an_cfg["buffer_files"]
    frac = an_cfg["abnormal_val_fraction"]
    a_n, n_n = len(abn_pos), len(normal_pos)
    v = int((a_n - b) * frac)
    t = a_n - b - v
    if v < 1 or t < 1:
        raise ValueError(f"{group}: only {a_n} abnormal files, too few to make validation and test sets")
    n_train = n_n - v - t - 2 * b
    if n_train < an_cfg["min_train_files"]:
        raise ValueError(f"{group}: only {n_train} normal files would remain for training "
                         f"(minimum {an_cfg['min_train_files']})")
    return {
        "val": list(abn_pos[:v]) + list(normal_pos[:v]),
        "test": list(abn_pos[a_n - t:]) + list(normal_pos[n_n - t:]),
        "train": list(normal_pos[v + b:n_n - t - b]),
        "buffer": list(abn_pos[v:v + b]) + list(normal_pos[v:v + b]) + list(normal_pos[n_n - t - b:n_n - t]),
    }


def _interleaved_split(normal_pos, abn_pos, an_cfg, group):
    """Spread validation and test over the WHOLE recording, so train, validation and test see the same
    sessions, loudness and fault types (a single block at each end would not).

    The abnormal sequence is cut into K chunks (K a multiple of 4). Chunks go to validation, test, test,
    validation, validation, test, test, ... so both sets reach early and late recordings. The first
    `buffer` files of a chunk are dropped whenever the previous chunk belongs to the other set. For each chunk, the SAME relative stretch of the
    normal sequence supplies exactly as many normal files, centred in the chunk with a gap on both sides; all other normal files
    are training files. So validation and test always have equal normal and abnormal counts."""
    b = an_cfg["buffer_files"]
    a_n, n_n = len(abn_pos), len(normal_pos)
    k = int(np.clip((a_n // 15) // 4 * 4, 4, an_cfg.get("interleaved_max_blocks", 16)))
    a_chunks = np.array_split(np.arange(a_n), k)
    n_chunks = np.array_split(np.arange(n_n), k)
    parts = {"val": [], "test": [], "train": [], "buffer": []}
    for j, (ac, nc) in enumerate(zip(a_chunks, n_chunks)):
        name = "val" if j % 4 in (0, 3) else "test"
        prev = "val" if (j - 1) % 4 in (0, 3) else "test"
        keep_a = ac[b:] if (j > 0 and prev != name) else ac   # gap only where val meets test
        parts["buffer"] += [abn_pos[i] for i in ac[:len(ac) - len(keep_a)]]
        parts[name] += [abn_pos[i] for i in keep_a]
        s = len(keep_a)
        if s + 2 * b > len(nc):
            raise ValueError(f"{group}: normal chunk {j} too short for {s} files plus gaps")
        pre = (len(nc) - s - 2 * b) // 2             # centre the evaluation files inside the chunk
        parts["train"] += [normal_pos[i] for i in nc[:pre]]
        parts["buffer"] += [normal_pos[i] for i in nc[pre:pre + b]]
        parts[name] += [normal_pos[i] for i in nc[pre + b:pre + b + s]]
        parts["buffer"] += [normal_pos[i] for i in nc[pre + b + s:pre + 2 * b + s]]
        parts["train"] += [normal_pos[i] for i in nc[pre + 2 * b + s:]]
    if min(len(parts["val"]), len(parts["test"])) // 2 < 5 or len(parts["train"]) < an_cfg["min_train_files"]:
        raise ValueError(f"{group}: too few files after interleaved split "
                         f"(train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])})")
    return parts


def _random_split(normal_pos, abn_pos, an_cfg, group, seed):
    rng = np.random.default_rng(_group_seed(seed, group))
    a_n, n_n = len(abn_pos), len(normal_pos)
    v = int(a_n * an_cfg["abnormal_val_fraction"])
    t = a_n - v
    if v < 1 or t < 1:
        raise ValueError(f"{group}: only {a_n} abnormal files, too few to make validation and test sets")
    if n_n - v - t < an_cfg["min_train_files"]:
        raise ValueError(f"{group}: too few normal files left for training")
    a = rng.permutation(np.array(abn_pos))
    n = rng.permutation(np.array(normal_pos))
    return {"val": list(a[:v]) + list(n[:v]), "test": list(a[v:]) + list(n[v:v + t]),
            "train": list(n[v + t:]), "buffer": []}


def assign_anomaly_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add column split_ad (train / val / test / buffer) for the normal-vs-abnormal detector.

    train = normal files only. val and test each contain as many normal as abnormal files."""
    out = df.copy()
    out["split_ad"] = "unused"
    an = cfg["split"]["anomaly"]
    for group, g in out.groupby("group"):
        normal = list(g[g["label"] == 0].sort_values("index_in_label").index)
        abn = list(g[g["label"] == 1].sort_values("index_in_label").index)
        if an["strategy"] == "interleaved":
            parts = _interleaved_split(normal, abn, an, group)
        elif an["strategy"] == "block":
            parts = _block_split(normal, abn, an, group)
        else:
            parts = _random_split(normal, abn, an, group, cfg["split"]["seed"])
        for name, idx in parts.items():
            out.loc[idx, "split_ad"] = name
    return out


# ----------------------------------------------------------------------------- classifier splits
def assign_classifier_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add column split_cls for fan-vs-pump: whole machines go to ONE split only."""
    cl = cfg["split"]["classifier"]
    mapping = {i: "train" for i in cl["train_ids"]}
    mapping.update({i: "val" for i in cl["val_ids"]})
    mapping.update({i: "test" for i in cl["test_ids"]})
    out = df.copy()
    out["split_cls"] = out["machine_id"].map(mapping).fillna("unused")
    for split in ("train", "val", "test"):
        present = set(out.loc[out["split_cls"] == split, "machine_type"])
        missing = set(cfg["dataset"]["machine_types"]) - present
        if missing:
            raise ValueError(f"classifier split '{split}' has no files for {sorted(missing)}")
    return out


# ----------------------------------------------------------------------------- checks
def validate_manifest(df: pd.DataFrame, cfg: dict) -> list:
    """Return a list of problems (an empty list means every leakage and labelling check passed)."""
    problems = []
    if df["file_id"].duplicated().any():
        problems.append("duplicate file_id values")
    if df["path"].duplicated().any():
        problems.append("duplicate paths")
    if not set(df["label"].unique()) <= {0, 1}:
        problems.append("labels other than 0/1 found")
    if (df["label_name"].map(LABELS) != df["label"]).any():
        problems.append("label does not match the folder name")

    if "split_ad" in df:
        if ((df["split_ad"] == "train") & (df["label"] == 1)).any():
            problems.append("an ABNORMAL file is in the anomaly training split")
        b = cfg["split"]["anomaly"]["buffer_files"]
        for group, g in df.groupby("group"):
            if "unused" in set(g["split_ad"]):
                problems.append(f"{group}: files without an anomaly split")
                continue
            for split in ("val", "test"):
                s = g[g["split_ad"] == split]
                if (s["label"] == 0).sum() != (s["label"] == 1).sum():
                    problems.append(f"{group}: {split} has unequal normal/abnormal counts")
            if cfg["split"]["anomaly"]["strategy"] in ("block", "interleaved"):
                # neighbouring recordings must never sit on both sides of a boundary:
                # file numbers in different splits must be at least buffer+1 apart
                for label in (0, 1):
                    gl = g[g["label"] == label]
                    idx = {s: gl.loc[gl["split_ad"] == s, "index_in_label"].to_numpy() for s in ("train", "val", "test")}
                    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
                        if len(idx[s1]) and len(idx[s2]):
                            gap = int(np.abs(idx[s1][:, None] - idx[s2][None, :]).min())
                            if gap < b + 1:
                                problems.append(f"{group}: {s1} and {s2} files are only {gap} apart (need more than {b})")

    if "split_cls" in df:
        by_machine = df.groupby("group")["split_cls"].nunique()
        if (by_machine > 1).any():
            problems.append("a machine appears in more than one classifier split")
        used = df[df["split_cls"] != "unused"]
        for split in ("train", "val", "test"):
            ids = set(used.loc[used["split_cls"] == split, "group"])
            for other in ("train", "val", "test"):
                if other != split and ids & set(used.loc[used["split_cls"] == other, "group"]):
                    problems.append("classifier splits share a machine")

    if "sha256" in df:
        dups = df[df["sha256"].astype(str).str.len() > 0].groupby("sha256")
        for sha, g in dups:
            if len(g) < 2:
                continue
            for col in ("split_ad", "split_cls"):
                if col in g and g[col].nunique() > 1:
                    problems.append(f"identical audio appears in different {col} values: {list(g['file_id'])[:3]}")
            if g["label"].nunique() > 1:
                problems.append(f"identical audio has different labels: {list(g['file_id'])[:3]}")
    return sorted(set(problems))
