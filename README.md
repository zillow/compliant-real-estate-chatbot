# compliant-real-estate-chatbot

## Installing required packages

We use `pip` to install the required packages. This project was tested with python 3.10 and pytorch 2.4.1. After installing 
them install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Reproducing data, models and results

### generating synthetic data

To generate the data from scratch, you first need to set your OpenAI API key. You can do this by exporting it as follow:

```bash
export OPENAI_API_KEY=your-api-key
```

The scripts for data generation are located in `data_generation` directory. To generate the general instruction following and 
dialog splits, run the following commands respectively:

```bash
python data_generation/diverse_QA_datagen.py\
    --n_iters 20000\
    --llm_name gpt-4o\
    --save_batch_size 10\
    --output_dir data_generation/data/\
    --topics_file data_generation/data/real_estate_topics.txt\
    --n_subtopics 50\
    --output_file_name general_instructions.json

python data_generation/diverse_conversation_datagen.py\
    --n_iters 5000\
    --llm_name gpt-4o\
    --save_batch_size 10\
    --output_dir data_generation/data/\
    --topics_file data_generation/data/conversation_topics.txt\
    --output_file_name dialogs.json
```

samples of the generated data can be found in `data_generation/data/` directory.

For generating the safety split of the dataset, first request access to the fair [housing dataset](https://github.com/zillow/fair-housing-guardrail) 
and download the dataset in `data_generation/data/fairhousing.json` directory. Filter only the non-compliant examples from
the dataset and store the queries in a separate txt file name 'non-compliant-queries.txt' (first section of `data_preparation.ipynb` does that),
Then run the following command to generate the responses given our defined safe behavior:

```bash
python data_generation/response_generator.py\
    --query_file data_generation/data/non-compliant-queries.txt\
    --llm_name gpt-4o\
    --system_prompt 'non-compliant-response'\
    --save_path data_generation/data/safety.json
```

You can then follow the rest of the `data_preparation.ipynb` notebook to postprocess the generated data, including conversion
to LLM chat format, pruning the dataset using `sentence_bert` transformer, and splitting the data into train, validation and test sets.