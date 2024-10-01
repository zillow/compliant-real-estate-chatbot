from fire import Fire
import os
import json
import glob
from data_generation.prompts import real_estate_chatbot_system_prompt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch


def generate_resp(model, tokenizer, examples, device, temperature=0.7, top_p=0.9, max_new_tokens=1024, do_sample=True):

    prompts = [[
        {"role": "system", "content": real_estate_chatbot_system_prompt},
        {"role": "user", "content": ex['query']},
    ] for ex in examples]

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
    return resps


def generate_responses(validation_dir: str = 'data/ft-v1', adaptor_path: str = 'outputs/', save_batch_size: int = 10,
                       model_short_name: str = "llama3-8b", model_response_col: str = "model_response", qlora=False):

    valid_data = []
    split_files = sorted(glob.glob(os.path.join(validation_dir, '*test.json')))
    for split_file in split_files:
        split_name = split_file.split('/')[-1].replace('_test.json', '')
        with open(split_file, 'r') as f:
            for line in f:
                ex = json.loads(line)
                if 'query' in ex:
                    valid_data.append({'query': ex['query'], 'response': ex['response'], 'split': split_name,
                                       'id': ex['id']})
                elif 'messages' in ex:
                    assert len(ex['messages']) == 2, f"Invalid example: {ex}"
                    valid_data.append({'query': ex['messages'][0]['content'], 'response': ex['messages'][1]['content'],
                                       'split': split_name, 'id': ex['id']})
                else:
                    raise ValueError(f"Invalid example: {ex}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("total validation data:", len(valid_data))

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
    lora_tokenizer.pad_token = "<|reserved_special_token_4|>"
    lora_tokenizer.pad_token_id = lora_tokenizer.convert_tokens_to_ids(lora_tokenizer.pad_token)

    lora_model.eval()

    result = []
    i = 0
    def checkpoint(data, save_path):
        with open(save_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save_path = os.path.join(validation_dir, f'{model_short_name}.json')

    pbar = tqdm(total=len(valid_data))
    while i < len(valid_data):
        examples = valid_data[i:i+save_batch_size]
        resps = generate_resp(lora_model, lora_tokenizer, examples, device)

        for ex, resp in zip(examples, resps):
            res_json = ex.copy()
            res_json[model_response_col] = resp
            result.append(res_json)

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)

        i += save_batch_size
        pbar.update(save_batch_size)

    checkpoint(result, save_path)


if __name__ == '__main__':
    Fire(generate_responses)