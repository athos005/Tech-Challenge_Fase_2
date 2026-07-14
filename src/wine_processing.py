import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuração de caminhos relativos (garante que funciona em qualquer computador)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def carregar_e_limpar_dados(caminho_arquivo):
    """Carrega o dataset e aplica a engenharia da variável alvo."""
    print("[1/4] A carregar os dados...")
    df = pd.read_csv(caminho_arquivo)

    # Remover a coluna Id, se existir
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Criar variável alvo (1 para qualidade >= 7, 0 caso contrário)
    df['is_high_quality'] = (df['quality'] >= 7).astype(int)

    return df

def preparar_dados(df):
    """Divide os dados e aplica a padronização das escalas."""
    print("[2/4] A separar variáveis de treino e teste...")
    X = df.drop(columns=['quality', 'is_high_quality'])
    y = df['is_high_quality']

    # Divisão com estratificação para manter a proporção das classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[3/4] A aplicar a padronização (StandardScaler)...")
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test

def executar_pipeline():
    """Executa o pipeline completo de pré-processamento."""
    caminho_csv = os.path.join(DATA_DIR, 'WineQT.csv')

    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Ficheiro não encontrado: {caminho_csv}. Verifique se está na pasta 'data/'.")

    df = carregar_e_limpar_dados(caminho_csv)
    X_train, X_test, y_train, y_test = preparar_dados(df)

    print("[4/4] A guardar os dados processados na pasta 'data/'...")
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)

    print("Pré-processamento concluído com sucesso!")

if __name__ == "__main__":
    executar_pipeline()
