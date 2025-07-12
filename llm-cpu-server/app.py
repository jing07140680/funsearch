import gc
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load model and tokenizer for GPT-2 small (or any smaller model)
model_name = "gpt2"  # You can use any smaller model like "gpt2", "distilgpt2", etc.
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to CPU (no GPU available)
device = "cpu"
model.to(device)

@app.route('/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt']
    repeat_prompt = content.get('repeat_prompt', 1)

    # Limit on repeat prompt to prevent excessive generation
    max_repeat_prompt = 10
    repeat_prompt = min(max_repeat_prompt, repeat_prompt)

    print(f'========================================== Prompt ==========================================')
    print(f'{prompt}\n')
    print(f'============================================================================================')
    print(f'\n\n')

    # Default parameters
    max_new_tokens = 512
    temperature = 0.3
    do_sample = True
    top_k = 30
    top_p = 0.9
    num_return_sequences = 1
    eos_token_id = 32021
    pad_token_id = 32021

    if 'params' in content:
        params: dict = content.get('params')
        max_new_tokens = params.get('max_new_tokens', 512)
        temperature = params.get('temperature', 0.3)
        do_sample = params.get('do_sample', True)
        top_k = params.get('top_k', 30)
        top_p = params.get('top_p', 0.9)
        num_return_sequences = params.get('num_return_sequences', 1)
        eos_token_id = params.get('eos_token_id', 32021)
        pad_token_id = params.get('pad_token_id', 32021)

    while True:
        # Tokenize the input without chat template
        inputs = tokenizer(prompt, return_tensors='pt')

        # Repeat the input for batch processing
        inputs = {key: value.repeat(repeat_prompt, 1).to(device) for key, value in inputs.items()}

        try:
            # LLM inference with max_new_tokens instead of max_length
            output = model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,  # Using max_new_tokens here
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
        except torch.cuda.OutOfMemoryError as e:
            # Handle out of memory error
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            repeat_prompt = max(repeat_prompt // 2, 1)
            continue

        # Decode and return response
        content = []
        for i in range(num_return_sequences):
            content.append(tokenizer.decode(output[i], skip_special_tokens=True))

        print(f'======================================== Response Content ========================================')
        print(f'{content}\n')
        print(f'==================================================================================================')
        print(f'\n\n')

        # Clear cache
        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        return jsonify({'content': content})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
