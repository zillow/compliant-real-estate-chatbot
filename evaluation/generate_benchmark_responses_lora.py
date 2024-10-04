import numpy as np
from fire import Fire
import os
import json
import glob
from data_generation.prompts import real_estate_chatbot_system_prompt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch


def generate_resp(model, tokenizer, examples, device, temperature=0.7, top_p=0.9, max_new_tokens=1024, do_sample=True):
    in_progress_examples = []
    processed_examples = []

    for ex in examples:
        in_progress_examples.append({
            'history': [],
            'query': ex['turns'][0],
            'follow_ups': ex['turns'][1:],
            'category': ex['category'],
            'session_id': ex['session_id'],
        })

    while len(in_progress_examples) > 0:

        prompts = []
        for ex in in_progress_examples:
            history = ex['history']
            prompts.append([
                {"role": "system", "content": real_estate_chatbot_system_prompt},
                *history,
                {"role": "user", "content": ex['query']},
            ])

        input_chat_prompts = [tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
        ) for prompt in prompts]

        inputs = tokenizer(input_chat_prompts, return_tensors='pt', padding=True, truncation=False)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        input_ids = inputs['input_ids'].to(device)
        attention_masks = inputs['attention_mask'].to(device)

        output = model.generate(
            input_ids,
            attention_mask=attention_masks,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

        resps = tokenizer.batch_decode(output[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=True)

        done_indices = []
        for i, (resp, ex) in enumerate(zip(resps, in_progress_examples)):
            context_str = []
            if len(ex['history']) > 0:
                context_str = [h['content'] for h in ex['history']]
            processed_examples.append({
                'query': ex['query'],
                'response': resp,
                'context_str': context_str,
                'category': ex['category'],
                'session_id': ex['session_id'],
            })
            ex['history'].append({"role": "user", "content": ex['query']})
            ex['history'].append({"role": "assistant", "content": resp})

            if len(ex['follow_ups']) > 0:
                ex['query'] = ex['follow_ups'][0]
                ex['follow_ups'] = ex['follow_ups'][1:]
            else:
                done_indices.append(i)

        in_progress_examples = [ex for i, ex in enumerate(in_progress_examples) if i not in done_indices]

    return processed_examples


def generate_responses(test_file: str = 'data/test_bench.json', adaptor_path: str = 'outputs/', save_batch_size: int = 10,
                       model_short_name: str = "llama3-lora", qlora=False):

    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            ex = json.loads(line)
            test_data.append(ex)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("total validation data:", len(test_data))

    if qlora:
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = True
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
        lora_model = AutoModelForCausalLM.from_pretrained(
            adaptor_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
        )
    else:
        lora_model = AutoModelForCausalLM.from_pretrained(adaptor_path)
        lora_model = lora_model.to(device)

    lora_tokenizer = AutoTokenizer.from_pretrained(adaptor_path)
    lora_tokenizer.padding_side = 'left'
    lora_tokenizer.pad_token = lora_tokenizer.eos_token
    lora_tokenizer.pad_token_id = lora_tokenizer.convert_tokens_to_ids(lora_tokenizer.pad_token)

    lora_model.eval()

    result = []
    i = 0
    def checkpoint(data, save_path):
        with open(save_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save_path = f'{test_file.replace(".json", "")}_{model_short_name}.json'

    pbar = tqdm(total=len(test_data))
    while i < len(test_data):
        examples = test_data[i:i+save_batch_size]
        processed_examples = generate_resp(lora_model, lora_tokenizer, examples, device)
        result.extend(processed_examples)

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)

        i += save_batch_size
        pbar.update(save_batch_size)

    checkpoint(result, save_path)


if __name__ == '__main__':
    Fire(generate_responses)