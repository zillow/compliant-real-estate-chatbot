import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, TextIteratorStreamer
from fire import Fire
from typing import List, Tuple
import torch
from transformers import BitsAndBytesConfig
from data_generation.prompts import real_estate_chatbot_system_prompt

def load_test_llama3_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_size = 256
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    model = LlamaForCausalLM(config=config)
    return model


system_prompt = real_estate_chatbot_system_prompt

def run(model_name_or_path='Qwen/Qwen1.5-0.5B-Chat', local_test=False, max_new_tokens=512,
        temperature=0.1, top_p=0.9, num_return_sequences=1, do_sample=True, num_beams=1,
        qlora=False, share=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    if local_test:
        model = load_test_llama3_model(model_name_or_path)
    elif qlora:
        print("Using QLora model")
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = model.to(device)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def build_input_from_chat_history(chat_history: List[Tuple], msg: str):
        messages = [{'role': 'system', 'content': system_prompt}]
        for user_msg, ai_msg in chat_history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        messages.append({'role': 'user', 'content': msg})
        return messages

    # Define the chat function
    def chat(message, chat_history):

        messages = build_input_from_chat_history(chat_history, message)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True
        )

        input_ids = input_ids.to(device)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                                do_sample=do_sample, temperature=temperature, top_p=top_p,
                                num_return_sequences=num_return_sequences, num_beams=num_beams)

        response_token_ids = output[0][input_ids.shape[1]:]
        response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
        chat_history.append((message, response))

        yield "", chat_history


    iface = gr.Blocks()

    with iface:
        gr.Markdown("# Let's Chat!")
        chatbox = gr.Chatbot([], layout="bubble", bubble_full_width=False)
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbox], [msg, chatbox])
        clear.click(lambda: None, None, chatbox, queue=False)

    # Launch the app
    iface.launch(share=share)


if __name__ == '__main__':
    Fire(run)