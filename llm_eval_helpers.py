# General
import inspect
#import logging
import random
import string
import warnings

from IPython.display import HTML, Markdown, display
import pandas as pd
from google.cloud import bigquery
import pandas as pd
# from google.cloud import storage


# Main
from vertexai.evaluation import EvalTask
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory

def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def print_doc(function):
    print(f"{function.__name__}:\n{inspect.getdoc(function)}\n")


def display_eval_report(eval_result, metrics=None):
    """Display the evaluation results."""

    title, summary_metrics, report_df = eval_result
    metrics_df = pd.DataFrame.from_dict(summary_metrics, orient="index").T
    if metrics:
        metrics_df = metrics_df.filter(
            [
                metric
                for metric in metrics_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
        report_df = report_df.filter(
            [
                metric
                for metric in report_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    # Display the title with Markdown for emphasis
    display(Markdown(f"## {title}"))

    # Display the metrics DataFrame
    display(Markdown("### Summary Metrics"))
    display(metrics_df)

    # Display the detailed report DataFrame
    display(Markdown("### Report Metrics"))
    display(report_df)
   
    temp_csv_file = "/tmp/report.csv"
    report_df.to_csv(temp_csv_file, index=False)
        
        # Upload CSV file to GCS
    # storage_client = storage.Client()
    # bucket = storage_client.get_bucket('llm-finetune-ndonthi1')
    # blob = bucket.blob('llm_eval')
    # blob.upload_from_filename(temp_csv_file)
        
    #print(f"File saved to GCS: gs://llm-finetune-ndonthi1/llm_eval")
    return report_df


def display_explanations(df, metrics=None, n=1):
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"
    df = df.sample(n=n)
    if metrics:
        df = df.filter(
            ["instruction", "context", "reference", "completed_prompt", "response"]
            + [
                metric
                for metric in df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    for index, row in df.iterrows():
        for col in df.columns:
            display(HTML(f"{col}: {row[col]}"))
        display(HTML(""))


def plot_radar_plot(eval_results, metrics=None):
    fig = go.Figure()

    for eval_result in eval_results:
        title, summary_metrics, report_df = eval_result

        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        fig.add_trace(
            go.Scatterpolar(
                r=list(summary_metrics.values()),
                theta=list(summary_metrics.keys()),
                fill="toself",
                name=title,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True
    )

    fig.show()


def plot_bar_plot(eval_results, metrics=None):
    fig = go.Figure()
    data = []

    for eval_result in eval_results:
        title, summary_metrics, _ = eval_result
        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group")
    fig.show()


def print_aggregated_metrics(job):
    """Print AutoMetrics"""

    rougeLSum = round(job.rougeLSum, 3) * 100
    display(
        HTML(
            f"The {rougeLSum}% of the reference summary is represented by LLM when considering the longest common subsequence (LCS) of words."
        )
    )


def print_autosxs_judgments(df, n=3):
    """Print AutoSxS judgments in the notebook"""

    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"
    df = df.sample(n=n)

    for index, row in df.iterrows():
        if row["confidence"] >= 0.5:
            display(
                HTML(
                    f"Document: {row['id_columns']['document']}"
                )
            )
            display(
                HTML(
                    f"Response A: {row['response_a']}"
                )
            )
            display(
                HTML(
                    f"Response B: {row['response_b']}"
                )
            )
            display(
                HTML(
                    f"Explanation: {row['explanation']}"
                )
            )
            display(
                HTML(
                    f"Confidence score: {row['confidence']}"
                )
            )
            display(HTML(""))


def print_autosxs_win_metrics(scores):
    """Print AutoSxS aggregated metrics"""

    score_b = round(scores["autosxs_model_b_win_rate"] * 100)
    display(
        HTML(
            f"AutoSxS Autorater prefers {score_b}% of time Model B over Model A "
        )
    )

