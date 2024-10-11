
# Main
from vertexai.evaluation import EvalTask
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory
from llm_eval_helpers import generate_uuid,display_eval_report
import json
import pandas as pd

def llm_eval(eval_dataset:str):
    data = json.loads(eval_dataset)
    
    # Create the pandas DataFrame from the JSON data
    eval_dataset = pd.DataFrame({
        "context": data["context"],
        "reference": data["reference"],
        "instruction": [data["instruction"]] * len(data["context"])
    })
  
    prompt_templates = [
    "Instruction: {instruction}. Article: {context}. Summary:",
    "Article: {context}. Complete this task: {instruction}, in one sentence. Summary:",
    "Goal: {instruction} and give me a TLDR. Here's an article: {context}. Summary:",
]
    
    metrics = [
    "rouge_1",
    "rouge_l_sum",
    "bleu",
    "fluency",
    "coherence",
    "safety",
    "groundedness",
    "summarization_quality",
    "verbosity",
]
    #Define EvalTask & Experiment
    generation_config = {
    "temperature": 0.3,
}

    safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    gemini_model = GenerativeModel(
        "gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    experiment_name = "eval-sdk-prompt-engineering"  # @param {type:"string"}

    summarization_eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=metrics,
        experiment=experiment_name,
    )

    run_id = generate_uuid()
    eval_results = []


    for i, prompt_template in enumerate(prompt_templates):
        experiment_run_name = f"eval-prompt-engineering-{run_id}-prompt-{i}"

        eval_result = summarization_eval_task.evaluate(
            prompt_template=prompt_template,
            experiment_run_name=experiment_run_name,
            model=gemini_model,
        )

        eval_results.append(
            (f"Prompt #{i}", eval_result.summary_metrics, eval_result.metrics_table)
        )
    return  eval_results   
    for eval_result in eval_results:
        display_eval_report(eval_result)

# llm_eval()