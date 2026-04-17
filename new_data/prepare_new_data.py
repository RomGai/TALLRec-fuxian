import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


INSTRUCTION = (
    "Given the user's preference history, identify whether the user will like the target item "
    'by answering "Yes." or "No.".'
)


def _read_user_rows(path: Path) -> List[Tuple[int, List[int], List[int]]]:
    rows: List[Tuple[int, List[int], List[int]]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_id, row in enumerate(reader, start=1):
            if len(row) != 3:
                print(f"[WARN] skip malformed row at {path}:{line_id}: {row}")
                continue
            user_id = int(row[0])
            pos = [int(x) for x in row[1].split(",") if x != ""]
            neg = [int(x) for x in row[2].split(",") if x != ""]
            if len(pos) < 2:
                print(f"[WARN] skip user {user_id} because pos length < 2")
                continue
            rows.append((user_id, pos, neg))
    return rows


def _load_item_text(item_desc_tsv: Path, i_map_tsv: Path) -> Dict[int, str]:
    item_text: Dict[int, str] = {}
    if item_desc_tsv.exists():
        with item_desc_tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                item_id_str = str(row.get("item_id", "")).strip()
                if not item_id_str.isdigit():
                    continue
                item_id = int(item_id_str)
                summary = (row.get("summary") or "").strip().replace("\n", " ")
                if summary:
                    item_text[item_id] = summary[:512]

    with i_map_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = int(row["item_id"])
            if item_id not in item_text:
                item_text[item_id] = f"Item_{item_id}"
    return item_text


def _item_to_text(item_text: Dict[int, str], item_id: int) -> str:
    txt = item_text.get(item_id, f"Item_{item_id}")
    return f'"{txt}"'


def _build_input(history: List[int], target: int, item_text: Dict[int, str], max_history: int) -> str:
    short_hist = history[-max_history:]
    hist_str = ", ".join(_item_to_text(item_text, x) for x in short_hist)
    target_str = _item_to_text(item_text, target)
    return f"User Preference History: {hist_str}\nWill the user like target item {target_str}?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TALLRec train/valid/test files from new_data format")
    parser.add_argument("--dataset", required=True, choices=["Baby_Products", "Video_Games"])
    parser.add_argument("--data-dir", default="new_data")
    parser.add_argument("--output-dir", default="prepared_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--neg-train-per-user", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=20)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / f"{args.dataset}_user_items_negs_train.csv"
    test_file = data_dir / f"{args.dataset}_user_items_negs_test.csv"
    i_map_file = data_dir / f"{args.dataset}_i_map.tsv"
    item_desc_file = data_dir / f"{args.dataset}_item_desc.tsv"

    print(f"[Step 1/6] Loading train rows from {train_file}")
    train_rows = _read_user_rows(train_file)
    print(f"[Step 2/6] Loading test rows from {test_file}")
    test_rows = _read_user_rows(test_file)

    train_user_ids = {x[0] for x in train_rows}
    test_user_ids = {x[0] for x in test_rows}
    overlap = train_user_ids & test_user_ids
    if overlap:
        print(f"[Leakage Fix] Found {len(overlap)} overlapping users, dropping them from training split")
        train_rows = [x for x in train_rows if x[0] not in overlap]

    print("[Step 3/6] Loading item text map")
    item_text = _load_item_text(item_desc_file, i_map_file)
    all_item_ids = sorted(item_text.keys())

    print("[Step 4/6] Splitting train/valid users")
    rng.shuffle(train_rows)
    split = int(len(train_rows) * (1 - args.val_ratio))
    train_split = train_rows[:split]
    valid_split = train_rows[split:]

    def build_train_examples(rows: List[Tuple[int, List[int], List[int]]]) -> List[dict]:
        examples: List[dict] = []
        for idx, (user_id, pos, negs) in enumerate(rows, start=1):
            history = pos[:-1]
            target = pos[-1]
            examples.append(
                {
                    "instruction": INSTRUCTION,
                    "input": _build_input(history, target, item_text, args.max_history),
                    "output": "Yes.",
                    "user_id": user_id,
                    "target_item_id": target,
                }
            )
            sampled_negs = negs[: args.neg_train_per_user]
            for neg in sampled_negs:
                examples.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": _build_input(history, neg, item_text, args.max_history),
                        "output": "No.",
                        "user_id": user_id,
                        "target_item_id": neg,
                    }
                )
            if idx % 1000 == 0:
                print(f"  [Train Convert] processed users={idx}/{len(rows)}")
        return examples

    print("[Step 5/6] Converting train/valid examples")
    train_examples = build_train_examples(train_split)
    valid_examples = build_train_examples(valid_split)

    test_records = []
    print("[Step 6/6] Building ranking test records")
    for idx, (user_id, pos, _negs) in enumerate(test_rows, start=1):
        history = pos[:-1]
        target = pos[-1]
        interacted = sorted(set(pos))
        test_records.append(
            {
                "user_id": user_id,
                "history": history,
                "target_item_id": target,
                "interacted_item_ids": interacted,
            }
        )
        if idx % 1000 == 0:
            print(f"  [Test Convert] processed users={idx}/{len(test_rows)}")

    (out_dir / "train.json").write_text(json.dumps(train_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "valid.json").write_text(json.dumps(valid_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "test_users.jsonl").open("w", encoding="utf-8") as f:
        for row in test_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "dataset": args.dataset,
        "num_train_users": len(train_split),
        "num_valid_users": len(valid_split),
        "num_test_users": len(test_records),
        "num_items": len(all_item_ids),
        "all_item_ids": all_item_ids,
        "seed": args.seed,
        "max_history": args.max_history,
    }
    (out_dir / "item_text.json").write_text(json.dumps(item_text, ensure_ascii=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Wrote:")
    print(f"  - {out_dir / 'train.json'}")
    print(f"  - {out_dir / 'valid.json'}")
    print(f"  - {out_dir / 'test_users.jsonl'}")
    print(f"  - {out_dir / 'item_text.json'}")
    print(f"  - {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
