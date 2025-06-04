import os
import fire
import logging
from termcolor import colored

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import load_dataset, disable_caching, Dataset, DatasetDict

from monkey_patch.fp8 import patch_create_accelerate_code_for_fp8
from callback import EfficiencyMLflowCallback

logger = logging.getLogger("mlflow").setLevel(logging.DEBUG)
transformers.logging.set_verbosity_info()
disable_caching()


def _load_dataset_any_type(data_path: str, split: str | None = None):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, split=split)
    elif data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files=data_path, split=split)
    elif data_path.endswith(".arrow"):
        data = Dataset.from_file(data_path)
    else:
        NotImplementedError(f"Your dataset type:{data_path} is not supported yet.")

    return data


def _convert_prompt_completion_format(example):
    assert (
        "instruction" in example and "output" in example
    ), "Each example must contain 'instruction' and 'output' fields."

    example["prompt"] = (
        example["instruction"] + " " + example["input"]
        if "input" in example and example ["input"]
        else example["instruction"]
    )
    # need space between prompt and completion
    # SFTTrainer simply concat both for tokenization
    # https://github.com/huggingface/trl/blob/cd6b3de356dd03025640681aaad7b45a1574d92f/trl/trainer/sft_trainer.py#L589
    example["completion"] = " " + example["output"]

    return example


def load_instruct_dataset(dataset_path: str):
    # load the instruct dataset in the prompt-completion format
    dataset = _load_dataset_any_type(dataset_path).map(
        _convert_prompt_completion_format,
        num_proc=os.cpu_count()
    )
    print(
        colored(f"[INFO] Instruct dataset loaded: {dataset}", "yellow")
    )
    return dataset


def get_lora_config(rank: int, alpha: float):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )
    

def monkey_patch_moe_block():
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from monkey_patch.fast_moe import ScatterMoeQwen3MoeSparseMoeBlock

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = ScatterMoeQwen3MoeSparseMoeBlock


def monkey_patch_sft_trainer():
    from monkey_patch.sft_trainer import patched_compute_loss

    SFTTrainer.compute_loss = patched_compute_loss


def main(
    model_path: str,
    tokenizer_path: str,
    dataset_path: str,
    output_dir: str = "",
    ## SFT hyperparameters ##
    train_steps: int = 100,
    save_steps: int = 500,
    global_batch_size: int = 128,
    micro_batch_size: int = 4,
    learning_rate: float = 2e-5,
    optim: str = "adamw_torch",
    deepspeed_config: str | None = None,
    fsdp: str = "",
    use_fp8: bool = False,
    weight_decay: float = 0.1,
    max_length: int = 4096,
    use_packing: bool = False,
    completion_only_loss: bool = False,
    torch_compile: bool = False,
    use_liger_kernel: bool = False,
    use_fast_moe_block: bool = False,
    full_determinism: bool = False,
    run_name: str = "",
    use_liger_efficiency_callback: bool = False,
    ## LoRA ##
    use_lora: bool = False,
    rank: int = 8,
    alpha: float = 0.5
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = load_instruct_dataset(dataset_path)

    if use_fast_moe_block:
        print(
            colored(f"[INFO] Activating fast MoE block", "yellow")
        )
        monkey_patch_moe_block()
    
    if use_liger_kernel:
        monkey_patch_sft_trainer()
    
    # zero 3 -> need to have training args before model init
    assert (
        (deepspeed_config or fsdp) and not (deepspeed_config and fsdp)
    ), "Cannot use DeepSpeed and FSDP at the same time!"

    gradient_accumulation_steps = global_batch_size // (micro_batch_size * torch.cuda.device_count())
    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=train_steps,
        learning_rate=learning_rate,
        gradient_checkpointing=not fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        optim=optim,
        deepspeed=deepspeed_config,
        fsdp=fsdp,
        fsdp_config="./config/fsdp.json" if fsdp else None,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=max_length,
        packing=use_packing,
        dataset_num_proc=os.cpu_count(),
        eos_token=tokenizer.eos_token,
        completion_only_loss=completion_only_loss,
        bf16=True,
        fp16=False,
        torch_compile=not use_liger_kernel, # NOTE: this will compile the whole model
        # use_liger=use_liger_kernel, # BUG: buggy in liger_kernel v0.5.10
        full_determinism=full_determinism,
        # report_to="mlflow",
        # run_name=run_name,
        include_num_input_tokens_seen=use_liger_efficiency_callback,
    )

    # load model
    # torchao FP8 is numerically better with flash attn?!
    # https://github.com/pytorch/ao/issues/556#issuecomment-2669442715
    model_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2" if use_packing or use_fp8 else "sdpa"
    )
    if use_liger_kernel:
        print(
            colored(
                "[INFO] Activating Liger Kernel "
                "and turn off its buggy LigerQwen3MoeSwiGLUMLP",
                "yellow"
            )
        )
        model = AutoLigerKernelForCausalLM.from_pretrained(
            swiglu=False,
            **model_kwargs
        )
        if torch_compile:
            torch.compile(model.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            **model_kwargs
        )
    # activate auxillary loss for load balancing
    model.config.output_router_logits = True
    
    if use_lora:
        lora_config = get_lora_config(rank, alpha)

    if use_fp8:
        print(
            colored(f"[INFO] Activate FP8 for training.", "yellow")
        )
        patch_create_accelerate_code_for_fp8()

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=(
            dataset["train"]
            if isinstance(dataset, DatasetDict) and "train" in dataset
            else dataset
        ),
        eval_dataset=(
            dataset["test"] 
            if isinstance(dataset, DatasetDict) and "test" in dataset 
            else None
        ),
        peft_config=lora_config if use_lora else None,
        processing_class=tokenizer,
        callbacks=(
            [
                EfficiencyMLflowCallback(
                    model_name=os.path.normpath(os.path.basename(model_path)), # qwen3
                    dataset_name=dataset_path.rstrip("/").split("/")[-3] # longalpaca
                )
            ]
            if use_liger_efficiency_callback
            else None
        )
    )
    trainer.train()

    if output_dir:
        trainer.save_model(output_dir)


if __name__ == '__main__':
    fire.Fire(main)