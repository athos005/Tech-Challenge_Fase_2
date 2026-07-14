import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Criar a pasta results se não existir
os.makedirs(RESULTS_DIR, exist_ok=True)

def carregar_dados_processados():
    """Carrega os dados previamente processados pelo preprocessamento.py"""
    print("A carregar dados processados...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))['is_high_quality']
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))['is_high_quality']
    return X_train, X_test, y_train, y_test

def treinar_e_avaliar(X_train, X_test, y_train, y_test):
    """Treina os modelos e gera as métricas."""
    print("A treinar os modelos (com balanceamento de classes)...")

    # Modelos
    log_model = LogisticRegression(class_weight='balanced', random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Previsões
    y_pred_log = log_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    # 1. Guardar o Relatório de Métricas em ficheiro de texto
    print("A exportar o relatório de métricas para a pasta results/...")
    caminho_metricas = os.path.join(RESULTS_DIR, 'metricas_modelos.txt')
    with open(caminho_metricas, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO: REGRESSÃO LOGÍSTICA ===\n")
        f.write(classification_report(y_test, y_pred_log, target_names=["Baixa/Média (0)", "Alta (1)"]))
        f.write("\n\n=== RELATÓRIO: RANDOM FOREST ===\n")
        f.write(classification_report(y_test, y_pred_rf, target_names=["Baixa/Média (0)", "Alta (1)"]))

    return rf_model, y_pred_log, y_pred_rf

def gerar_graficos(rf_model, X_train, y_test, y_pred_log, y_pred_rf):
    """Gera e guarda os gráficos na pasta results/"""
    print("A gerar e guardar os gráficos...")

    # --- 1. Gráfico de Matrizes de Confusão ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_log = confusion_matrix(y_test, y_pred_log)
    disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Baixa/Média", "Alta"])
    disp_log.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Matriz de Confusão - Regressão Logística')
    axes[0].grid(False)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Baixa/Média", "Alta"])
    disp_rf.plot(ax=axes[1], cmap='Greens', colorbar=False)
    axes[1].set_title('Matriz de Confusão - Random Forest')
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'matrizes_confusao.png'), dpi=300)
    plt.close()

    # --- 2. Gráfico de Feature Importance (Random Forest) ---
    importances = rf_model.feature_importances_
    df_importances = pd.DataFrame({'Feature': X_train.columns, 'Importância': importances})
    df_importances = df_importances.sort_values(by='Importância', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importância', y='Feature', data=df_importances, palette='mako')
    plt.title('Importância das Variáveis Físico-Químicas na Qualidade do Vinho', fontsize=14)
    plt.xlabel('Grau de Importância Relativa')
    plt.ylabel('Componente Químico')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'importancia_variaveis.png'), dpi=300)
    plt.close()

    print("Sucesso! Todos os resultados foram guardados na pasta 'results/'.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = carregar_dados_processados()
    rf_model, y_pred_log, y_pred_rf = treinar_e_avaliar(X_train, X_test, y_train, y_test)
    gerar_graficos(rf_model, X_train, y_test, y_pred_log, y_pred_rf)
