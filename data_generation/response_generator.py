import os

import fire
import json
from langchain_openai import ChatOpenAI

from tqdm import tqdm
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import real_estate_chatbot_system_prompt, non_compliant_response_system_prompt
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_community.callbacks import get_openai_callback


def generate_data(query_file, system_prompt='non-compliant-response', llm_name='gpt-4o', save_batch_size=1,
                  save_path=None, temperature=0.7, max_tokens=2048):

    if 'gpt' in llm_name:
        llm = ChatOpenAI(model_name=llm_name, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE,
                         default_headers={"providers": "openai"}, max_tokens=max_tokens, temperature=temperature,)
    else:
        raise ValueError("LLM not implemented")

    print("using llm: ", llm_name)

    if save_path is None:
        save_path = query_file.replace('queries', 'responses')
        save_path = save_path.replace('.txt', '.json')

    queries = []
    with open(query_file, 'r') as f:
        data = f.read().strip()
        for line in data.split('\n'):
            queries.append(line.strip())

    print("Total queries: ", len(queries))

    print("using system prompt: ", system_prompt)
    if system_prompt == 'general':
        system_prompt = real_estate_chatbot_system_prompt
    elif system_prompt == 'non-compliant-response':
        system_prompt = non_compliant_response_system_prompt
    else:
        raise ValueError("System prompt not implemented")

    results = []
    i = 0

    pbar = tqdm(total=len(queries))
    with get_openai_callback() as cb:
        while i < len(queries):
            try:
                batch = queries[i:i+save_batch_size]

                messages_list = [[
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query),
                ] for query in batch]

                chain = llm | StrOutputParser()
                resps = chain.batch(messages_list)

                for query, resp in zip(batch, resps):
                    print("query: ", query)
                    print("response: ", resp)
                    print("-"*100)

                    results.append({
                        'query': query,
                        'response': resp
                    })

                with open(save_path, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')

                i += save_batch_size
                pbar.update(save_batch_size)

            except Exception as e:
                time.sleep(5*60)
                continue

        print('total cost: ', cb.total_cost)


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')

    fire.Fire(generate_data)
