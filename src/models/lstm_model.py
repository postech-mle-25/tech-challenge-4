from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class StockLSTM:
    def __init__(self, input_shape, units=[128, 64, 32]):
        self.input_shape = input_shape
        self.units = units
        self.model = None
        
    def build_model(self):
        """Constrói arquitetura LSTM"""
        model = Sequential()
        
        # Primeira camada LSTM
        model.add(LSTM(
            units=self.units[0],
            return_sequences=True,
            input_shape=self.input_shape
        ))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Segunda camada LSTM
        model.add(LSTM(units=self.units[1], return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Terceira camada LSTM
        model.add(LSTM(units=self.units[2], return_sequences=False))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Camadas densas
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1))
        
        # Compilar
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Treina o modelo"""
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history