import contextlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Literal

import transformers
from transformers import Trainer

from alpaca_farm import common, constants, data_utils, logging, utils

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default=None)
    dataset_name: str = field(default="sft")
    eval_size: int = field(
        default=500,
        metadata={"help": "Number of examples to split out from training to use for evaluation."},
    )
    prompt_dict_path: str = field(
        default="prompts/sft_prompt.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded to this length (and possibly truncated)."
            "Enforcing a consistent max length ensures memory usage is constant and predictable."
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if training_args.deepspeed is not None:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    elif training_args.initialize_model_on_cpu:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = True
    else:
        ctx_mgr = common.staggered_object_creation(
            local_rank=training_args.local_rank, world_size=training_args.world_size
        )
        device_map = {"": training_args.device.index}
        low_cpu_mem_usage = True

    with ctx_mgr:
        model: transformers.PreTrainedModel = common.make_generative_lm(
            model_name_or_path=model_args.model_name_or_path,
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            config=transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
        )
        common.let_model_save_mem_when_zero_grad(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",  # Ensures properly masking out the source tokens.
        use_fast=training_args.use_fast_tokenizer,
    )
    tokenizer.padding = training_args.padding

    # Collect special tokens. Only add if non-existent.
    special_tokens_dict = dict(additional_special_tokens=[])
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = training_args.pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = constants.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = constants.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = constants.DEFAULT_UNK_TOKEN
    utils.stable_resize_token_embeddings_and_tokenizer(model, tokenizer, special_tokens_dict)

    data_module: dict = data_utils.make_ShareGPTsupervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Tokenizer is only supplied so that it gets saved; this makes loading easier.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.warning("hooray! training finished successfully! now on to model saving.", main_process_only=True)

    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    logger.warning("hooray again! model saving worked.", main_process_only=True)


if __name__ == "__main__":
    main()
