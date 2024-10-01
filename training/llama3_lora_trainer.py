from dataclasses import dataclass, field
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, LlamaForCausalLM, AutoConfig, EarlyStoppingCallback
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
import nltk
from data_generation.prompts import real_estate_chatbot_system_prompt
import glob
from peft import LoraConfig

from trl import (
    SFTConfig,
    SFTTrainer)



@dataclass
class ScriptArguments:
    train_file: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    dev_dir: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "Model ID to use for SFT training"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load the model in 4bit"}
    )
    use_peft: bool = field(
        default=False, metadata={"help": "Whether to use PEFT"}
    )
    flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use fast attention"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Path to the cache directory"}
    )
    num_return_sequences: int = field(
        default=1, metadata={"help": "Number of sequences to generate"}
    )
    local_test: bool = field(
        default=False, metadata={"help": "Whether to run a local test"}
    )
    max_new_tokens: int = field(
        default=100, metadata={"help": "Max number of tokens to generate"}
    )
    eval_do_sample: bool = field(
        default=True, metadata={"help": "Whether to sample during evaluation"}
    )
    eval_temperature: float = field(
        default=0.3, metadata={"help": "Temperature for sampling during evaluation"}
    )
    eval_max_samples: int = field(
        default=None, metadata={"help": "Max number of samples to evaluate from each dev file"}
    )
    eval_generation_batch_size: int = field(
        default=64, metadata={"help": "Batch size for generation during evaluation"}
    )
    eval_top_p: float = field(
        default=0.93, metadata={"help": "Top p for sampling during evaluation"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "Alpha for LoRA"}
    )
    lora_r: int = field(
        default=64, metadata={"help": "R for LoRA"}
    )


def load_test_llama3_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_size = 256
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    model = LlamaForCausalLM(config=config)
    return model


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "json",
        data_files=script_args.train_file,
        split="train",
        cache_dir=script_args.cache_dir,
    )

    dev_datasets = {}
    dev_files = glob.glob(os.path.join(script_args.dev_dir, "*val.json"))
    print(f"list of validation files: {dev_files}")

    if len(dev_files) > 0:
        for dev_file_path in dev_files:
            print('loading ', dev_file_path)
            ds = load_dataset(
                "json",
                data_files=dev_file_path,
                split="train",
                cache_dir=script_args.cache_dir,
            )
            dev_split_name = dev_file_path.split("/")[-1].split(".")[0]
            if script_args.eval_max_samples is not None:
                ds = ds.select(range(script_args.eval_max_samples))
            dev_datasets[dev_split_name] = ds
    else:
        raise NotImplementedError("No dev files provided")

    def add_system_prompt(example):
        # mistral doesn't support system messages

        if 'mistral' not in script_args.model_id.lower():
            example["messages"].insert(0, {"role": "system", "content": real_estate_chatbot_system_prompt})

        if example['messages'][0]['role'] == 'assistant':
            example['messages'] = example['messages'][1:]

        return example

    train_dataset = train_dataset.map(add_system_prompt)

    for k, dev_dataset in dev_datasets.items():
        dev_datasets[k] = dev_dataset.map(add_system_prompt)


    ################
    # Model & Tokenizer
    ###############
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True, cache_dir=script_args.cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    device_map = None
    if 'llama-3' in script_args.model_id.lower() and not script_args.local_test:
        print("Using llama 3 recommended data type")
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = True
    elif 'llama-2' in script_args.model_id.lower() and not script_args.local_test:
        print("Using llama 2 recommended data type")
        torch_dtype = torch.float16
        quant_storage_dtype = torch.float16
        bnb_4bit_use_double_quant = False
        device_map = {"": 0}
    elif 'mistral' in script_args.model_id.lower() and not script_args.local_test:
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = None
    else:
        torch_dtype = None
        quant_storage_dtype = None
        bnb_4bit_use_double_quant = None

    quantization_config = None
    if script_args.load_in_4bit:
        print("Using 4bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    if not script_args.local_test:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            quantization_config=quantization_config,
            use_flash_attention_2=script_args.flash_attention,
            torch_dtype=quant_storage_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            cache_dir=script_args.cache_dir,
            device_map=device_map)
    else:
        model = load_test_llama3_model(script_args.model_id)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.05,
            r=script_args.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=training_args.max_seq_length,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        packing=True,
    )

    if trainer.accelerator.is_main_process and script_args.use_peft:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    print("training_args: ", training_args)
    print("script_args: ", script_args)
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)