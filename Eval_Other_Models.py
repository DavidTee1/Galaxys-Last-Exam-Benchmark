import pandas as pd
import json
import openai
import time
from tqdm import tqdm  

with open("config.json", "r") as f:
    config = json.load(f)

OPENAI_API_KEY = config.get("openai")
openai.api_key = OPENAI_API_KEY

print("Loading data...")
#you may need to change dir here
results_df = pd.read_csv("results.csv")
total_rows = len(results_df)
print(f"Loaded {total_rows} rows from results.csv")

def evaluate_answer(prompt, solution, model_answer):
    """
    Uses the gpt-4o-mini model to evaluate if the model_answer matches the expected solution.
    Returns 1 if correct, 0 if incorrect.
    """
    evaluation_prompt = (
        f"Evaluate the following answer against the expected solution.\n\n"
        f"Prompt: {prompt}\n\n"
        f"Expected Solution: {solution}\n\n"
        f"Model Answer: {model_answer}\n\n"
        f"Respond with a single number: 1 if the model answer is correct, or 0 if it is incorrect."
    )
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0,
            )
         
            result_text = response.choices[0].message.content.strip()

            if "1" in result_text:
                return 1
            else:
                return 0
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt 
                print(f"Error during API call: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return 0

#modify this for addtl models
model_columns = [
    "o1",
    "gpt-4o", 
    "gpt-4o-mini", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro", 
    "claude-3-7-sonnet-20250219"
]

existing_model_columns = [col for col in model_columns if col in results_df.columns]
if len(existing_model_columns) < len(model_columns):
    missing = set(model_columns) - set(existing_model_columns)
    print(f"Warning: The following columns were not found in the data: {missing}")

total_evaluations = total_rows * len(existing_model_columns)
completed_evaluations = 0

print(f"\nStarting evaluation of {len(existing_model_columns)} models across {total_rows} samples...")
print(f"Total operations: {total_evaluations}\n")

master_pbar = tqdm(total=total_evaluations, desc="Overall Progress", position=0)

for model_idx, col in enumerate(existing_model_columns):
    print(f"\nEvaluating model {model_idx+1}/{len(existing_model_columns)}: {col}")
    
    evaluation_results = []
    
    model_pbar = tqdm(total=total_rows, desc=f"{col}", position=1, leave=False)
    
    for index, row in results_df.iterrows():
        if pd.isna(row[col]):
            evaluation_results.append(0)
            completed_evaluations += 1
            master_pbar.update(1)
            model_pbar.update(1)
            continue
        #the results.csv doesn't natively generate solutions from the original prompts - paste these over, or modify the script    
        eval_result = evaluate_answer(row['Prompt'], row['Solution'], row[col])
        evaluation_results.append(eval_result)
        
        completed_evaluations += 1
        master_pbar.update(1)
        model_pbar.update(1)
    
    model_pbar.close()
    
    results_df[f"eval_{col}"] = evaluation_results
    
    interim_columns = ["Prompt", "Solution"] + [col] + [f"eval_{col}"]
    results_df[interim_columns].to_csv(f"interim_eval_{col}.csv", index=False)
    
    correct = sum(evaluation_results)
    accuracy = correct / total_rows
    print(f"Model {col} results: {correct}/{total_rows} correct ({accuracy:.2%})")

master_pbar.close()

print("\nSaving final results...")
output_columns = ["Prompt", "Solution"] + existing_model_columns + [f"eval_{col}" for col in existing_model_columns]
results_df[output_columns].to_csv("evaluation_matrix.csv", index=False)

summary = {}
for col in existing_model_columns:
    eval_col = f"eval_{col}"
    correct = results_df[eval_col].sum()
    accuracy = correct / total_rows
    summary[col] = {"correct": int(correct), "total": total_rows, "accuracy": accuracy}

summary_df = pd.DataFrame(summary).T
summary_df.to_csv("evaluation_summary.csv")

print("\n===== EVALUATION SUMMARY =====")
for model, stats in summary.items():
    print(f"{model}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")
print("===============================")

print("\nEvaluation complete. Results saved to:")
print("- evaluation_matrix.csv (detailed results)")
print("- evaluation_summary.csv (performance summary)")