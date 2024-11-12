import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pathlib import Path


def generate_trace(df, model_name, metric, headline_var, color):
    """Generate a single trace for a model, metric, and variable."""
    y = df[headline_var].to_numpy().squeeze()
    return go.Scatter(
        x=np.arange(1, df.shape[0] + 1),
        y=y,
        mode='lines',
        name=f"{model_name} vs. ERA5",
        customdata=[f"{metric}-{headline_var}"],
        line=dict(width=3, color=color),
        visible=(metric == 'rmse' and headline_var == 't-850')
    )


def update_visibility(fig, selected_metric, selected_variable):
    """Update trace visibility based on selected metric and variable."""
    selected_combo = f"{selected_metric}-{selected_variable}"
    return [
        trace.customdata[0] == selected_combo
        for trace in fig.data
    ]


def create_dropdown_buttons(metrics, headline_vars, fig):
    """Create dropdown buttons for metric-variable combinations."""
    return [
        {
            "method": "update",
            "label": f"{metric.upper()} - {var.capitalize()}",
            "args": [
                {"visible": update_visibility(fig, metric, var)}
            ],
        }
        for metric in metrics
        for var in headline_vars.keys()
    ]


def configure_layout(fig, title, metrics, headline_vars):
    """Configure layout and dropdown menu for the figure."""
    fig.update_layout(
        updatemenus=[{
            "buttons": create_dropdown_buttons(metrics, headline_vars, fig),
            "direction": "down",
            "showactive": True,
            "x": 0,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
            "name": "Metric-Variable"
        }],
        title=title,
        xaxis_title="Number of Days Ahead",
        hovermode="x unified"
    )


def save_figure(fig, output_dir, filename):
    """Save the figure as an interactive HTML file."""
    output_path = output_dir / filename
    fig.write_html(output_path)


def plot_metrics(metrics, headline_vars, model_names, data_path, output_dir, title, filename, ensemble=False):
    """Generate interactive plots for the given metrics, models, and variables."""
    fig = go.Figure()
    linecolors = [
        'black', '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b', '#e377c2'
    ]

    for metric in metrics:
        for model_idx, model_name in enumerate(model_names):
            color = linecolors[model_idx % len(linecolors)]
            for headline_var in headline_vars:
                if ensemble:
                    csv_path = data_path / f"{model_name}_ensemble/eval/{metric}_{model_name}.csv"
                else:
                    csv_path = data_path / f"{model_name}/eval/{metric}_{model_name}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    fig.add_trace(generate_trace(df, model_name, metric, headline_var, color))

    configure_layout(fig, title, metrics, headline_vars)
    save_figure(fig, output_dir, filename)


def main():
    """
    Main driver to generate interactive HTML for metrics display
    Usage example: `python compute_climatology.py --dataset_name era5 --is_spatial 0`
    """
    
    output_dir = Path('../website/html')
    output_dir.mkdir(parents=True, exist_ok=True)

    control_model_names = [
        'climatology', 'panguweather', 'graphcast', 'fourcastnetv2',
        'ecmwf', 'ncep', 'ukmo', 'cma',
    ]
    ensemble_model_names = ['ecmwf', 'ncep', 'ukmo', 'cma']
    
    headline_vars = {'t-850': 'K', 'z-500': 'gpm', 'q-700': 'g/kg'}
    
    # Control (deterministic metrics) plot
    plot_metrics(
        metrics=['rmse', 'acc'],
        headline_vars=headline_vars,
        model_names=control_model_names,
        data_path=Path('../logs'),
        output_dir=output_dir,
        title="Control (Deterministic Metrics)",
        filename="control.html"
    )

    # Ensemble (probabilistic metrics) plot
    plot_metrics(
        metrics=['rmse', 'crpss'],
        headline_vars=headline_vars,
        model_names=ensemble_model_names,
        data_path=Path('../logs'),
        output_dir=output_dir,
        title="Ensemble (Probabilistic Metrics)",
        filename="ensemble.html",
        ensemble=True
    )


if __name__ == "__main__":
    main()
