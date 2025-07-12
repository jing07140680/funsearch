import time
import requests
import json

url = 'http://127.0.0.1:5000/completions'  # Adjusted to match the Flask server's port

while True:
    prompt = '''Complete a Python function inspired by the given functions. 
You must generate a function that is different from the previous functions.
You must be creative and you can add various control flow statements. 
Only output the Python code, no descriptions. 
'''

    data = {
        'prompt': prompt,
        'repeat_prompt': 5,
        'system_prompt': '',
        'stream': False,
        'params': {
            'temperature': 0.5,
            'top_k': None,
            'top_p': None,
            'add_special_tokens': False,
            'skip_special_tokens': True,
        }
    }

    headers = {'Content-Type': 'application/json'}

    # Time the request to check the response duration
    record_time = time.time()
    response = requests.post(url, data=json.dumps(data), headers=headers)
    durations = time.time() - record_time

    def process_response_content(content: str) -> str:
        # Process the response content to extract the generated Python function
        ret = content.split('[/INST]')[1]
        return ret

    if response.status_code == 200:
        print(f'Query time: {durations:.2f} seconds')
        content = response.json()["content"]
        for c in content:
            print(f'Generated Function: {c}')
    else:
        print('Failed to make the POST request.')
