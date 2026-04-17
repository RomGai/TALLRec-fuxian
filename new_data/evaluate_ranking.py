import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = (
    "Given the user's preference history, identify whether the user will like the target item "
    'by answering "Yes." or "No.".'
)


def _build_prompt_plain(history: List[int], target: int, item_text: Dict[str, str], max_history: int) -> str:
    hist = history[-max_history:]
    hist_text = ", ".join(f'"{item_text.get(str(i), f"Item_{i}")}"' for i in hist)
    target_text = f'"{item_text.get(str(target), f"Item_{target}")}"'
    input_text = f"User Preference History: {hist_text}\nWill the user like target item {target_text}?"
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    )


def _running_metrics(ranks: List[int], k: int) -> Dict[str, float]:
    hr = sum(1 for r in ranks if r <= k) / max(1, len(ranks))
    ndcg = sum((1 / math.log2(r + 1)) if r <= k else 0.0 for r in ranks) / max(1, len(ranks))
    return {"hr": hr, "ndcg": ndcg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranking evaluation with 1 target + 1000 random negatives")
    parser.add_argument("--base-model", required=True, help="HF model id/path")
    parser.add_argument("--lora-weights", default="", help="Optional LoRA weights path")
    parser.add_argument("--prepared-dir", required=True, help="Directory created by prepare_new_data.py")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--neg-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--output", default="ranking_metrics.json")
    parser.add_argument("--max-users", type=int, default=-1, help="Only evaluate first N users when > 0")
    parser.add_argument("--prompt-style", default="auto", choices=["auto", "plain", "chat"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Step 1/5] Loading tokenizer/model from Hugging Face. device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if args.lora_weights:
        print(f"[Step 1.1] Loading LoRA weights from {args.lora_weights}")
        model = PeftModel.from_pretrained(model, args.lora_weights)
    model.eval()
    use_chat_template = (
        (args.prompt_style == "chat")
        or (args.prompt_style == "auto" and getattr(tokenizer, "chat_template", None))
    )
    print(f"[Prompt Style] use_chat_template={bool(use_chat_template)}")

    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    if len(yes_ids) == 0:
        raise RuntimeError("Cannot tokenize ' Yes'; please verify tokenizer.")
    yes_token_id = yes_ids[0]

    prepared_dir = Path(args.prepared_dir)
    item_text = json.loads((prepared_dir / "item_text.json").read_text(encoding="utf-8"))
    meta = json.loads((prepared_dir / "meta.json").read_text(encoding="utf-8"))
    all_item_ids = meta["all_item_ids"]
    test_rows = [json.loads(x) for x in (prepared_dir / "test_users.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_users > 0:
        test_rows = test_rows[: args.max_users]

    print(f"[Step 2/5] Loaded {len(test_rows)} users, catalog size={len(all_item_ids)}")

    ranks: List[int] = []

    print("[Step 3/5] Start per-user ranking evaluation")
    tic = time.time()
    for user_idx, row in enumerate(test_rows, start=1):
        user_id = row["user_id"]
        history = row["history"]
        target = row["target_item_id"]
        interacted = set(row["interacted_item_ids"])

        candidate_pool = [x for x in all_item_ids if x not in interacted and x != target]
        if len(candidate_pool) < args.neg_sample_size:
            sampled_negs = candidate_pool
        else:
            user_rng = random.Random((args.seed * 1000003) + int(user_id))
            sampled_negs = user_rng.sample(candidate_pool, args.neg_sample_size)

        candidates = [target] + sampled_negs
        if use_chat_template:
            prompts = []
            for item in candidates:
                hist = history[-args.max_history:]
                hist_text = ", ".join(f'"{item_text.get(str(i), "Item_" + str(i))}"' for i in hist)
                target_text = f'"{item_text.get(str(item), "Item_" + str(item))}"'
                user_text = (
                    "Given the user's preference history, identify whether the user will like the target item "
                    "by answering \"Yes.\" or \"No.\".\n\n"
                    f"User Preference History: {hist_text}\n"
                    f"Will the user like target item {target_text}?"
                )
                messages = [
                    {"role": "system", "content": "You are a helpful recommender assistant."},
                    {"role": "user", "content": user_text},
                ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        else:
            prompts = [_build_prompt_plain(history, item, item_text, args.max_history) for item in candidates]

        scores: List[float] = []
        for st in range(0, len(prompts), args.batch_size):
            ed = min(st + args.batch_size, len(prompts))
            batch_prompts = prompts[st:ed]
            model_inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
            with torch.no_grad():
                outputs = model(**model_inputs)
                logits = outputs.logits
                last_pos = model_inputs["attention_mask"].sum(dim=1) - 1
                batch_idx = torch.arange(logits.size(0), device=logits.device)
                next_token_logits = logits[batch_idx, last_pos, :]
                prob_yes = torch.softmax(next_token_logits, dim=-1)[:, yes_token_id]
                scores.extend(prob_yes.float().cpu().tolist())

            print(
                f"  [User {user_idx}/{len(test_rows)}] step batch={st // args.batch_size + 1} "
                f"processed_candidates={ed}/{len(prompts)}"
            )

        target_score = scores[0]
        rank = 1 + sum(1 for s in scores[1:] if s > target_score)
        ranks.append(rank)

        m10 = _running_metrics(ranks, 10)
        m20 = _running_metrics(ranks, 20)
        m40 = _running_metrics(ranks, 40)
        progress = (user_idx / max(1, len(test_rows))) * 100
        elapsed = time.time() - tic
        print(
            f"[User Done] {user_idx}/{len(test_rows)} ({progress:.2f}%) "
            f"user_id={user_id} rank={rank} elapsed={elapsed:.1f}s | "
            f"avg_HR@10={m10['hr']:.4f} avg_NDCG@10={m10['ndcg']:.4f} | "
            f"avg_HR@20={m20['hr']:.4f} avg_NDCG@20={m20['ndcg']:.4f} | "
            f"avg_HR@40={m40['hr']:.4f} avg_NDCG@40={m40['ndcg']:.4f}"
        )

    print("[Step 4/5] Aggregating final metrics")
    result = {
        "num_users": len(ranks),
        "HR@10": _running_metrics(ranks, 10)["hr"],
        "NDCG@10": _running_metrics(ranks, 10)["ndcg"],
        "HR@20": _running_metrics(ranks, 20)["hr"],
        "NDCG@20": _running_metrics(ranks, 20)["ndcg"],
        "HR@40": _running_metrics(ranks, 40)["hr"],
        "NDCG@40": _running_metrics(ranks, 40)["ndcg"],
        "mean_rank": sum(ranks) / max(1, len(ranks)),
        "seed": args.seed,
        "neg_sample_size": args.neg_sample_size,
    }

    print("[Step 5/5] Saving metrics")
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
