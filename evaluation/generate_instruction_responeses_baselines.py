import numpy as np
from fire import Fire
import os
import json
from typing import List, Dict
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
import glob
from data_generation.prompts import real_estate_chatbot_system_prompt
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from utils import load_hf_embedding_model
from langchain_openai import ChatOpenAI


def generate_responses(validation_dir: str = 'data/ft-v1', llm_name: str = 'meta.llama3-8b-instruct-v1:0',
                       n_shots: int = 10, train_file: str = "data/ft-v1/v1-train.json", save_batch_size: int = 10,
                       model_short_name: str = "llama3-8b-10shot",
                       model_response_col: str = "model_response",
                       embedding_model="sentence-transformers/all-mpnet-base-v2"):
    if 'llama' in llm_name or 'mistral' in llm_name:
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

    if n_shots > 0:
        train_examples = []
        with open(train_file, 'r') as f:
            for line in f:
                train_examples.append(json.loads(line))

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

    print("total validation data:", len(valid_data))
    if n_shots == 0:
        prompt_tmp = ChatPromptTemplate.from_messages([
            SystemMessage(content=real_estate_chatbot_system_prompt),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
    else:
        train_examples = []

        with open(train_file, 'r') as f:
            for line in f:
                train_examples.append(json.loads(line))

        # for test:
        # np.random.shuffle(train_examples)
        # train_examples = train_examples[:10]

        train_few_shot_template = ChatPromptTemplate.from_messages(
            [("human", "{query}"), ("ai", "{response}")],
        )

        to_vectorize = []
        metadatas = []

        for ex in train_examples:
            if 'query' in ex:
                to_vectorize.append(ex['query'])
                metadatas.append(ex)
            elif 'messages' in ex:
                if len(ex['messages']) == 0:
                    print("WARNING: empty message")
                    continue
                to_vectorize.append(ex['messages'][0]['content'])
                metadatas.append({'query': ex['messages'][0]['content'], 'response': ex['messages'][1]['content']})
            else:
                raise ValueError(f"Invalid example: {ex}")

        vectorstore = Chroma.from_texts(to_vectorize, load_hf_embedding_model(embedding_model), metadatas=metadatas)

        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=n_shots
        )

        few_shot_prompts = FewShotChatMessagePromptTemplate(
            # The input variables select the values to pass to the example_selector
            example_selector=example_selector,
            example_prompt=train_few_shot_template,
        )

        prompt_tmp = ChatPromptTemplate.from_messages([
            SystemMessage(content=real_estate_chatbot_system_prompt),
            few_shot_prompts,
            HumanMessagePromptTemplate.from_template("{query}")
        ])

    chain = prompt_tmp | llm | StrOutputParser()

    def checkpoint(data, save_path):
        with open(save_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save_path = os.path.join(validation_dir, f'{model_short_name}.json')

    result = []
    for ex in tqdm(valid_data):
        # print(prompt_tmp.invoke({'query': ex['query']}))
        resp = chain.invoke({'query': ex['query']})
        res_json = ex.copy()
        res_json[model_response_col] = resp
        result.append(res_json)

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)
    checkpoint(result, save_path)


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
    Fire(generate_responses)