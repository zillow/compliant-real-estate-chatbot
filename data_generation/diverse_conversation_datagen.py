import os
import time
import fire
from datetime import datetime
import numpy as np
import json
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

def generate_data(n_iters=5, llm_name='gpt-4o', save_batch_size=1, output_dir='data/ft-v6',
                  topics_file='data/conversation_topics.txt', temperature=0.7, max_tokens=4096,
                  output_file_name='dialogs.json'):

    os.makedirs(output_dir, exist_ok=True)

    with open(topics_file, 'r') as f:
        topic_list = f.read().strip().split('\n')

    print("Total topic_list: ", len(topic_list))

    llm = ChatOpenAI(
        model_name=llm_name,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        max_tokens=max_tokens,
        temperature=temperature,
        default_headers={"providers": "openai"}
    )

    template = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""Your task is to generate a comprehensive and helpful conversation\
 between two parties. Assume that a user is chatting with a real estate chatbot. FIRST, assume the topic of the\
 conversation is "{topic}" and write 50 possible scenarios of conversation in a numbered list (just title is enough).\
 SECOND choose scenario {N} and state it. THIRD, generate a complete and long conversation between the two parties.\
 Assistant's utterances should be long and helpful. At the beginning of the conversation write "<Conversation>".\
 begin assistant utterances with "Assistant:" and user utterances\
 with "User:" user should start the conversation. Be creative.""")])


    chain = template | llm | StrOutputParser()

    results = []
    cost = 0
    output_path = os.path.join(output_dir, output_file_name)
    pbar = tqdm(total=n_iters)

    i = 0
    while i < n_iters:
        topics = np.random.choice(topic_list, size=save_batch_size, replace=False)
        Ns = np.random.randint(low=1, high=51, size=save_batch_size)

        try:
            with get_openai_callback() as cb:
                resps = chain.batch([{'topic': topic, 'N': n} for topic, n in zip(topics, Ns)])
                cost += cb.total_cost
            print("Accumulated cost: ", cost)
        except Exception as e:
            print(e)
            time.sleep(5*60)
            output_path = os.path.join(output_dir, f'{output_file_name}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json')
            continue

        for resp, topic in zip(resps, topics):
            print("-"*100)
            print("topic:", topic)
            print("response:", resp)

            results.append({
                'topic': topic,
                'response': resp,
            })

        i += save_batch_size
        pbar.update(save_batch_size)
        print("total cost: ", cost)
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')

    fire.Fire(generate_data)
