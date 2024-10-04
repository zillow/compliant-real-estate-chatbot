import torch
from fire import Fire
import os
import json
from langchain_core.prompts import (PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate,
                                    FewShotChatMessagePromptTemplate, HumanMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from data_generation.prompts import real_estate_chatbot_system_prompt
from tqdm import tqdm
from langchain_openai import ChatOpenAI


def generate_resp(llm, examples):
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

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", real_estate_chatbot_system_prompt),
        ("placeholder", "{history}"),
        ("user", "{query}")
    ])

    chain = prompt_template | llm | StrOutputParser()

    while len(in_progress_examples) > 0:

        prompts = []
        for ex in in_progress_examples:
            prompts.append({
                "history": ex['history'],
                "query": ex['query']
            })

        resps = chain.batch(prompts)

        done_indices = []
        for i, (resp, ex) in enumerate(zip(resps, in_progress_examples)):
            context_str = []
            if len(ex['history']) > 0:
                context_str = [h[1] for h in ex['history']]
            processed_examples.append({
                'query': ex['query'],
                'response': resp,
                'context_str': context_str,
                'category': ex['category'],
                'session_id': ex['session_id'],
            })
            ex['history'].append(("user", ex['query']))
            ex['history'].append(("assistant", resp))

            if len(ex['follow_ups']) > 0:
                ex['query'] = ex['follow_ups'][0]
                ex['follow_ups'] = ex['follow_ups'][1:]
            else:
                done_indices.append(i)

        in_progress_examples = [ex for i, ex in enumerate(in_progress_examples) if i not in done_indices]

    return processed_examples


def generate_responses(test_file: str = 'data/test_bench.json', llm_name: str = 'meta.llama3-8b-instruct-v1:0',
                       save_batch_size: int = 10,
                       model_short_name: str = "llama3-8b"):

    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            ex = json.loads(line)
            test_data.append(ex)

    print("total validation data:", len(test_data))

    if 'llama' in llm_name:
        # llama3_name = 'meta.llama3-8b-instruct-v1:0'
        llm = ChatBedrock(
            model_id=llm_name,
            model_kwargs={"temperature": 0.7, "max_gen_len": 1024},
        )
    elif 'gpt' in llm_name:
        llm = ChatOpenAI(model_name=llm_name, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE,
                         temperature=0.7, max_tokens=1024, default_headers={"providers": "openai"})
    else:
        raise ValueError("LLM not implemented")


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
        processed_examples = generate_resp(llm, examples)
        result.extend(processed_examples)

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)

        i += save_batch_size
        pbar.update(save_batch_size)

    checkpoint(result, save_path)


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    Fire(generate_responses)