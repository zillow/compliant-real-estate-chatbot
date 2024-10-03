import os
import time
import fire
from datetime import datetime
import numpy as np
import json
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from prompts import real_estate_chatbot_system_prompt
from langchain_community.callbacks import get_openai_callback
import re


def generate_data(n_iters=5, llm_name='gpt-4o', save_batch_size=1, output_dir='data/ft-v6',
                  topics_file='data/ft-v4/real_estate_topics.txt', n_subtopics=50,
                  temperature=0.7, max_tokens=2048, output_file_name='general_instructions.json'):

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

    subtopic_and_question_generator_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""First write {num_subtopics} subtopics about "{topic}" topic that you can answer questions about. Then\
 pick subtopic {N}. Second state the chosen subtopic. Third, write a single question that is not about the chosen subtopic but can only be answered with\
 expertise in real estate domain and in that subtopic. You must begin your questions with "Question:" without any formatting's.\
 Be creative and write a challenging question.""")])

    def question_extractor(response):
        response = response.strip().lower()
        pattern = r'question: ([^\n]+)'
        matches = list(re.finditer(pattern, response))

        if len(matches) == 0:
            return ""
        question = matches[-1].group(1).strip()
        return question


    question_gen_chain = subtopic_and_question_generator_prompt | llm | StrOutputParser()

    template = ChatPromptTemplate.from_messages([
        SystemMessage(real_estate_chatbot_system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    chain = template | llm | StrOutputParser()

    results = []
    cost = 0
    output_path = os.path.join(output_dir, output_file_name)
    pbar = tqdm(total=n_iters)

    i = 0
    while i < n_iters:
        topics = np.random.choice(topic_list, size=save_batch_size, replace=False)
        Ns = np.random.randint(low=1, high=n_subtopics+1, size=save_batch_size)

        try:
            with get_openai_callback() as cb:
                subtopic_and_questions = question_gen_chain.batch([{'topic': topic, 'N': n, 'num_subtopics': n_subtopics}
                                                                   for topic, n in zip(topics, Ns)])
                questions = [question_extractor(resp) for resp in subtopic_and_questions]
                resps = chain.batch([{'question': q} for q in questions])
                cost += cb.total_cost
            print("Accumulated cost: ", cost)
        except Exception as e:
            print(e)
            time.sleep(5*60)
            output_path = os.path.join(output_dir, f'{output_file_name}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json')
            continue

        for resp, topic, int, question in zip(resps, topics, subtopic_and_questions, questions):
            print("-"*100)
            print("topic:", topic)
            print("question:", question)
            print("response:", resp)

            if len(question) < 10:
                print("Skipping question because it's too short")
                continue

            results.append({
                'topic': topic,
                'subtopics': int,
                'response': resp,
                'query': question
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
