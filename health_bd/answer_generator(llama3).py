import json
import requests

# --- Configuration ---
LLAMA3_API_URL = "http://localhost:11434/api/generate"  # Adjust if your endpoint is different (e.g., /api/chat)
MODEL_NAME = "llama3"  # Replace with the exact model name you pulled (e.g., "llama3:8b")
INPUT_FILE = "Q.jsonl" # This is your specified input file name
OUTPUT_FILE = "generated_answers_llama3_to_mistral-7B.json" # <--- Changed extension to .json
MAX_TOKENS = 300  # Adjust as needed for response length
TEMPERATURE = 0.7 # Adjust for creativity (0.0 for deterministic, 1.0 for more creative)
# Fixed values for the output structure
RESPONSED_MODEL_NAME = "Llama-3" # Changed to Llama-3 as requested
# PROMPT_TYPE_FOR_OUTPUT is not in the desired output, so it's commented out/removed from output_entry

def get_llama3_response(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
    }
    try:
        response = requests.post(LLAMA3_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Llama3 API for prompt: '{prompt[:50]}...' - {e}")
        return None

def process_questions(input_file, output_file):
    responses = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile):
            try:
                question_data = json.loads(line.strip())
                
                # Extract data based on the new input structure
                question_no = question_data.get("question_no", line_num + 1)
                data_chuck = question_data.get("data_chuck", None)
                model_q = question_data.get("model(q)", "unknown") # Using model(q) from input
                prompt = question_data.get("question") 

                if not prompt:
                    print(f"Skipping line {line_num + 1} due to missing 'question' key: {line.strip()}")
                    continue

                print(f"Processing question {question_no} (Chuck {data_chuck}): '{prompt[:70]}...'")
                llama_response_raw = get_llama3_response(prompt)

                generated_text = ""
                if llama_response_raw:
                    generated_text = llama_response_raw.get("response", "No 'response' key found in Llama3 output.")
                else:
                    generated_text = "Error: Could not get response from Llama3."
                
                # Construct output_entry according to the desired JSON structure
                output_entry = {
                    "question_no": question_no,
                    "data_chuck": data_chuck,
                    "model(q)": model_q, # Retaining the model(q) from the input
                    "question": prompt,
                    "Responsed_model": "Llama3", # Using the new fixed value
                    "model_response": generated_text,
                }
                responses.append(output_entry)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num + 1}: {line.strip()} - {e}")
            except Exception as e:
                print(f"An unexpected error occurred on line {line_num + 1}: {line.strip()} - {e}")

    # --- THIS IS THE CHANGE FOR JSON OUTPUT ---
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(responses, outfile, indent=4, ensure_ascii=False)
    # --- END OF CHANGE ---

    print(f"\nProcessing complete. Responses saved to {output_file}")

if __name__ == "__main__":
    process_questions(INPUT_FILE, OUTPUT_FILE)
