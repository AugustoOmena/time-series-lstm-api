import pickle
import torch
import torch.nn as nn
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import traceback
from app.schemas.ticker_request import TickerRequestBetweenDates, TickerRequest
from typing import Tuple, List
from app.config.settings import get_settings

settings = get_settings()

# Definição do Modelo LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # input_size=5 ('Close', 'daily_return', '5-day_volatility', '10-day_volatility', '15-day_volatility')
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Camanda dropout (desliga neuronios)
        self.dropout = nn.Dropout(dropout_prob)

        # Conecta a saída do LSTM ao output final (previsão de 1 valor)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass for the SimpleLSTM.
        Expects input x with shape (batch, seq_len, input_size).
        Returns tensor with shape (batch, output_size).
        """
        # LSTM returns output for all timesteps and the hidden state tuple
        out, (hn, cn) = self.lstm(x)
        # Use the last timestep's output for prediction
        last = out[:, -1, :]
        last = self.dropout(last)
        out = self.fc(last)
        return out


# CPU - Custom Unpickler para forçar o carregamento na CPU
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        if module == '__main__':
            if name in globals():
                return globals()[name]

        # Para o resto, usa o comportamento padrão
        return super().find_class(module, name)
        
# Construindo a janela deslizante
def create_sequences_multivariate(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def obtemDadosHistoricos(ticker, data_inicial, data_final):
    dados= yf.download(ticker, start=data_inicial, end=data_final)
    colunas= []
    for col in dados.columns:
        colunas.append(col[0])
    dados.columns= colunas
    return dados

# Estratégia 2: Preço de abertura, máxima, mínima, fechamento e volume
def build_features_estrategia2(data):
    columns = ['Close']
    return data[columns]


def getX_testY_test_Sliding_Window(command: TickerRequestBetweenDates):
    # 1. Converter string para data real
    dt_inicial = pd.to_datetime(command.init_date)
    
    # 2. Calcular o "Buffer" de segurança.
    # Precisamos de 30 dias ÚTEIS para trás. 
    # Como existem fins de semana, pegamos 60 dias corridos para garantir que sobe.
    buffer_dias = settings.SEQ_LENGTH * 2 
    dt_fetch_start = dt_inicial - pd.Timedelta(days=buffer_dias)
    
    # 3. Baixar dados com margem extra
    end_date_adjusted = pd.to_datetime(command.end_date) + pd.Timedelta(days=1)
        
    dados_brutos = obtemDadosHistoricos(
        command.ticker, 
        dt_fetch_start, 
        end_date_adjusted.strftime('%Y-%m-%d')
    )
    
    # Resetar index para facilitar manipulação se o indice for data
    if isinstance(dados_brutos.index, pd.DatetimeIndex):
        dados_brutos = dados_brutos.reset_index()
        
    # Localizar o índice da primeira data >= data_inicial
    # Coluna 'Date' geralmente é criada pelo reset_index ou yfinance
    mask_start = dados_brutos.iloc[:, 0] >= dt_inicial
    if not mask_start.any():
         raise ValueError("Data inicial não encontrada nos dados baixados.")
         
    idx_start_user = mask_start.idxmax()
    
    # O corte deve começar SEQ_LENGTH posições antes desse índice
    idx_corte = idx_start_user - settings.SEQ_LENGTH
    
    if idx_corte < 0:
        print("AVISO: Histórico insuficiente para cobrir a janela completa antes da data inicial.")
        idx_corte = 0

    # Cortamos o dataframe para começar exatamente onde precisamos
    dados_validos = dados_brutos.iloc[idx_corte:]

    if 'Date' in dados_validos.columns:
        todas_datas = dados_validos['Date'].to_numpy()
    else:
        todas_datas = dados_validos.index.to_numpy()
    
    dados_validos = dados_validos.set_index(dados_validos.columns[0])
    
    data = build_features_estrategia2(dados_validos)

    # Construindo a janela deslizante
    data_np = data.to_numpy()
    
    # Verificação de segurança
    if len(data_np) <= settings.SEQ_LENGTH:
         raise ValueError(f"Dados insuficientes ({len(data_np)}) para janela de {settings.SEQ_LENGTH}.")

    X, y = create_sequences_multivariate(data_np, settings.SEQ_LENGTH)

    datas_y = todas_datas[settings.SEQ_LENGTH : settings.SEQ_LENGTH + len(y)]

    X_test_reshaped = X.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_test_reshaped)

    X_test_norm = scaler.transform(X_test_reshaped).reshape(X.shape)
    y_test_norm = scaler.transform(y.reshape(-1, 1))

    # Convertendo para tensores
    X_test = torch.from_numpy(X_test_norm).float().to(settings.DEVICE).unsqueeze(-1)
    y_test = torch.from_numpy(y_test_norm).float().to(settings.DEVICE).unsqueeze(1)
    
    return (X_test, y_test, scaler, datas_y)



def obtemX_para_um_dia(command: TickerRequest):
    """
    Versão corrigida usando comparação de strings para garantir match da data.
    """
    target_dt = pd.to_datetime(command.target_date).normalize()
    target_str = target_dt.strftime('%Y-%m-%d')
    
    # Busca com margem maior (5 dias) para garantir fins de semana/feriados
    end_fetch = target_dt + pd.Timedelta(days=5) 
    start_fetch = target_dt - pd.Timedelta(days=settings.SEQ_LENGTH * 2 + 10) 

    dados = obtemDadosHistoricos(command.ticker, start_fetch.date().isoformat(), end_fetch.date().isoformat())
    
    if dados.empty:
         return None, None, None, {"error": f"Nenhum dado encontrado para {command.ticker}"}

    data_processed = build_features_estrategia2(dados)

    # CRIAMOS UMA LISTA DE STRINGS PARA BUSCA SEGURA
    # Isso ignora completamente se o índice é UTC, Naive, etc.
    datas_disponiveis = [d.strftime('%Y-%m-%d') for d in data_processed.index]
    
    actual_price = None
    seq = None

    if command.target_date.strftime("%Y-%m-%d") in datas_disponiveis:
        # CENÁRIO A: Encontramos a data exata (Dia útil passado/presente fechado)
        # Pegamos a posição inteira (índice numérico) onde a string bate
        idx_target = datas_disponiveis.index(target_str)
        
        actual_price = float(data_processed.iloc[idx_target, 0])
        
        # Pega a sequência dos 30 dias ANTERIORES a esse índice
        seq = data_processed.iloc[idx_target - settings.SEQ_LENGTH : idx_target].to_numpy()
        
    else:
        # CENÁRIO B: Data futura ou dia sem pregão
        # Pegamos os últimos 30 dias disponíveis do dataframe
        actual_price = None
        seq = data_processed.iloc[-settings.SEQ_LENGTH:].to_numpy()

    if len(seq) < settings.SEQ_LENGTH:
        return None, None, None, {
            "error": f"Histórico insuficiente. Temos {len(seq)}, precisamos de {settings.SEQ_LENGTH}."
        }

    # Montagem do Tensor
    X = seq.reshape(1, settings.SEQ_LENGTH, seq.shape[1])
    
    X_reshaped = X.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_reshaped)
    X_norm = scaler.transform(X_reshaped).reshape(X.shape)

    X_test = torch.from_numpy(X_norm).float().to(settings.DEVICE).unsqueeze(-1)

    return X_test, scaler, actual_price, None

def run_forecast(model, dados_tensor):
    """
    Retorna uma tupla (resultado, erro).
    Se houver erro, 'resultado' é None e 'erro' contém o dict pronto para retorno.
    """
    try:
        with torch.no_grad():
            prediction = model(dados_tensor.squeeze(3)).cpu().numpy()
            return prediction, None
            
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Erro interno no PyTorch: {e}")
        
        error_response = {
            "error": "Erro durante inferência", 
            "details": str(e), 
            "trace": tb
        }

        return None, error_response


def generate_recursive_forecast(
    model, 
    scaler, 
    last_window_tensor: torch.Tensor, 
    last_val_norm: float,
    last_date: any, 
    target_end_date: str
) -> Tuple[List[any], List[float]]:
    
    # 1. Normalização de datas para evitar erro de Timezone/Horas
    dt_last = pd.to_datetime(last_date).normalize()
    dt_target = pd.to_datetime(target_end_date).normalize()

    if dt_target <= dt_last:
        print("AVISO: Data alvo é anterior ou igual à última data. Retornando vazio.")
        return [], []

    # 2. Preparação da Janela Inicial
    # Pega o tensor [30, 1] ou [30, 1, 1]
    current_window = last_window_tensor.unsqueeze(0)
    
    if current_window.dim() == 4:
        current_window = current_window.squeeze(-1)

    new_point = torch.tensor([[[last_val_norm]]], device=current_window.device, dtype=torch.float32)

    # Remove o dia mais velho (index 0) e insere o último valor conhecido no final
    current_window = torch.cat((current_window[:, 1:, :], new_point), dim=1)

    # 3. Geração de Datas Futuras (Dias Úteis)
    dates_range = pd.date_range(start=dt_last + pd.Timedelta(days=1), end=dt_target, freq='B')

    future_dates = []
    future_preds = []

    model.eval()
    
    for future_date in dates_range:
        with torch.no_grad():
            # Inferência
            pred_norm_tensor = model(current_window)
            pred_norm = pred_norm_tensor.item()
            
            # Desnormalização para salvar
            # scaler.inverse_transform espera array 2D
            pred_real = scaler.inverse_transform([[pred_norm]])[0][0]
            
            future_dates.append(future_date)
            future_preds.append(pred_real)
            
            # Atualiza Janela para o próximo dia
            new_point_loop = torch.tensor([[[pred_norm]]], device=current_window.device, dtype=torch.float32)
            current_window = torch.cat((current_window[:, 1:, :], new_point_loop), dim=1)

    return future_dates, future_preds