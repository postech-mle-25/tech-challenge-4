import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, sequence_length=60, train_split=0.8):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, df, target_col='Close'):
        """Prepara dados para LSTM"""
        
        # Selecionar features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[features].values
        
        # Normalizar
        scaled_data = self.scaler.fit_transform(data)
        
        # Criar sequências
        X, y = self.create_sequences(scaled_data, target_col_idx=3)
        
        # Split train/test
        split_idx = int(len(X) * self.train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def create_sequences(self, data, target_col_idx):
        """Cria sequências temporais para LSTM"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_col_idx])
            
        return np.array(X), np.array(y)