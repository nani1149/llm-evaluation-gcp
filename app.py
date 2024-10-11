import gradio as gr
import json
import pandas as pd
import plotly.graph_objects as go
from llm_eval_helpers import generate_uuid
from llm_eval import llm_eval


# Modified display_eval_report function to return DataFrames
def display_eval_report(eval_result, metrics=None):
    """Return the evaluation results as DataFrames."""
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
    
    return title, metrics_df, report_df

# Updated plot_bar_plot function to return a Plotly figure
def plot_bar_plot(eval_results, metrics=None):
    fig = go.Figure()
    data = []

    # Create bar plot from eval_results
    for eval_result in eval_results:
        title, summary_metrics, _ = eval_result
        
        # Filter metrics if any specific ones are requested
        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        # Create a bar for each evaluation result
        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    # Create a grouped bar chart
    fig = go.Figure(data=data)

    # Update layout for grouped bar mode
    fig.update_layout(
        barmode="group",
        title="Evaluation Metrics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Values",
        legend_title="Evaluation Titles"
    )

    return fig  # Return the Plotly figure

# Function to evaluate and plot the results
def greet(json_input: str):
    eval_results = llm_eval(json_input)
    
    # Use plot_bar_plot function to generate the bar plot
    fig = plot_bar_plot(eval_results)
    
    return fig  # Return the figure for Gradio to display

# Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs="text",   # Expecting JSON string input
    outputs="plot",  # Return the bar plot
)


demo.launch()