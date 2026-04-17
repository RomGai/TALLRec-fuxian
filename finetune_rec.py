import os
import sys
import inspect
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from transformers import BitsAndBytesConfig

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F402
from sklearn.metrics import roc_auc_score

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    load_in_8bit: bool = True,
    logging_steps: int = 1,
    prompt_style: str = "auto",  # auto | plain | chat

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"load_in_8bit: {load_in_8bit}\n"
        f"logging_steps: {logging_steps}\n"
        f"prompt_style: {prompt_style}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    print("[Step 1/8] Loading backbone model")
    model_load_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": device_map,
    }
    if load_in_8bit:
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_load_kwargs,
        )
    except TypeError as e:
        if load_in_8bit:
            print(
                "[Warning] 8-bit load failed due to Transformers/PEFT compatibility. "
                "Retrying without 8-bit quantization.\n"
                f"Original error: {e}"
            )
            model_load_kwargs.pop("quantization_config", None)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                **model_load_kwargs,
            )
        else:
            raise

    print("[Step 2/8] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Allow batched inference
    yes_token_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_token_ids = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes_token_ids) == 0 or len(no_token_ids) == 0:
        raise ValueError("Tokenizer cannot tokenize ' Yes' / ' No'.")
    yes_token_id = yes_token_ids[0]
    no_token_id = no_token_ids[0]
    print(f"[Token IDs] yes_token_id={yes_token_id}, no_token_id={no_token_id}")

    use_chat_template = (
        (prompt_style == "chat")
        or (prompt_style == "auto" and getattr(tokenizer, "chat_template", None))
    )
    print(f"[Prompt Style] use_chat_template={bool(use_chat_template)}")

    def build_prompt(data_point, with_output=True):
        if use_chat_template:
            user_text = f"{data_point['instruction']}\n\n{data_point['input']}".strip()
            messages = [
                {"role": "system", "content": "You are a helpful recommender assistant."},
                {"role": "user", "content": user_text},
            ]
            if with_output:
                messages.append({"role": "assistant", "content": data_point["output"]})
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not with_output,
            )
        return generate_prompt(data_point if with_output else {**data_point, "output": ""})

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = build_prompt(data_point, with_output=True)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = build_prompt(data_point, with_output=False)
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    print("[Step 3/8] Preparing model for PEFT/LoRA")
    if load_in_8bit and "quantization_config" in model_load_kwargs:
        model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    

    print("[Step 4/8] Loading train/validation datasets")
    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)
    train_count = len(train_data["train"])
    val_count = len(val_data["train"])
    print(f"[Data Check] train examples={train_count}, valid examples={val_count}")
    if train_count == 0:
        print(
            "[Warning] Loaded 0 training examples at this stage. "
            "If this is transient in your environment, training may continue after dataset generation."
        )
    if val_count == 0:
        print(
            "[Warning] Validation set has 0 examples. "
            "Consider lowering --val-ratio in prepare_new_data.py."
        )

    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    print("[Step 5/8] Tokenizing training dataset")
    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    print("[Step 6/8] Tokenizing validation dataset")
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    def compute_metrics(eval_preds):
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        return {'auc': auc}
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        labels_index = torch.argwhere(torch.bitwise_or(labels == yes_token_id, labels == no_token_id))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == no_token_id, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:, [no_token_id, yes_token_id]], dim=-1)
        return logits[:, 1][2::3], gold[2::3]

    os.environ["WANDB_DISABLED"] = "true"
    
    if sample > -1:
        if sample <= 128:
            eval_step = 10
        else:
            eval_step = int(sample / 128 * 5)
    else:
        eval_step = 100

    class VerboseStepCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"[Train Log] step={state.global_step} logs={logs}")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            print(f"[Eval] step={state.global_step} metrics={metrics}")
    
    print("[Step 7/8] Building Trainer")
    training_args_kwargs = dict(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        optim="adamw_torch",
        save_strategy="steps",
        eval_steps=eval_step,
        save_steps=eval_step,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=None,
    )
    ta_sig = inspect.signature(transformers.TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        training_args_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_sig.parameters:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        raise RuntimeError(
            "Neither 'evaluation_strategy' nor 'eval_strategy' is supported by this transformers version."
        )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(**training_args_kwargs),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10), VerboseStepCallback()],
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("[Step 8/8] Starting training loop")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print(f"[Done] Saved LoRA adapter to {output_dir}")

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
