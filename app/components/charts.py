import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix_plotly(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Normal', 'Spoofed'],
        y=['Normal', 'Spoofed'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False
    ))
    fig.update_layout(
        title=f'{model_name} Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=400
    )
    return fig

def plot_roc_curve_plotly(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f'{model_name} ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        width=500,
        height=400,
        showlegend=True
    )
    return fig

