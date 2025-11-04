# src/utils/metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de avaliação"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def plot_predictions(y_true, y_pred, dates=None):
    """Visualiza previsões vs valores reais"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    if dates is not None:
        ax.plot(dates, y_true, label='Real', color='blue')
        ax.plot(dates, y_pred, label='Previsto', color='red', alpha=0.7)
    else:
        ax.plot(y_true, label='Real', color='blue')
        ax.plot(y_pred, label='Previsto', color='red', alpha=0.7)
    
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Preço')
    ax.set_title('Previsões vs Valores Reais')
    ax.legend()
    plt.show()