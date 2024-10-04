# sample of running your fine-tuned model
python generate_instruction_responses_lora.py\
 --validation_dir data/validation_dir\
 --adaptor_path models/your_finetuned_lora_path\
 --model_short_name llama3_lora\
 --save_batch_size 16

# sample of running llama3 baselines from aws bedrock
python generate_instruction_responses_baselines.py\
 --llm_name meta.llama3-70b-instruct-v1:0\
 --validation_dir data/validation_dir\
 --n_shots 5\
 --train_file data/train.json\
 --save_batch_size 10\
 --model_short_name llama3-70b-5shot


# sample of running openai baselines
OPENAI_API_KEY="your api key here" python generate_instruction_responses_baselines.py\
 --llm_name gpt-4\
 --validation_dir data/validation_dir\
 --n_shots 5\
 --train_file data/train.json\
 --save_batch_size 10\
 --model_short_name gpt-4-5shot