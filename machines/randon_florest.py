import os
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import joblib

def treinar_modelo(csv_path, modelo_path=os.path.join(os.getcwd(), "machines", "modelos", "randon_florest_modelo.joblib")):
    # 1) carregar dados
    df = pd.read_csv(csv_path)

    # checagem de colunas
    cols_req = ["N","P","K","temperature","humidity","ph","rainfall","indice"]
    faltando = [c for c in cols_req if c not in df.columns]
    if faltando:
        raise ValueError(f"Faltam colunas no CSV: {faltando}")

    # 2) separar X (features) e y (alvo)
    X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
    y = df["indice"]

    # 3) treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) modelo
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 5) avaliação
    y_pred = rf.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    # rmse compatível com versões antigas
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"R²  : {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")

    # 6) importâncias
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nImportância das variáveis:")
    print(importances.round(3))

    # 7) salvar modelo
    joblib.dump(rf, modelo_path)
    print(f"\nModelo salvo em: {modelo_path}")

    return rf, importances

def usar_modelo(N, P, K, temperature, humidity, ph, rainfall,
                 modelo_path = os.path.join(os.getcwd(), "machines", "modelos", "randon_florest_modelo.joblib"),
                 multiplicar_por_60=False):
    # carregar modelo
    rf = joblib.load(modelo_path)

    # montar DataFrame com uma linha
    X = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # prever índice
    indice_pred = float(rf.predict(X)[0])
    resultado = {"indice_pred": round(indice_pred, 3)}

    # aplicar regra: se índice < 0.33 → inviável, zera produção
    if multiplicar_por_60:
        if indice_pred < 0.33:
            resultado["sacas_est"] = 0.0
        else:
            resultado["sacas_est"] = round(indice_pred * 60.0, 1)

    # print mais limpo
    print(resultado)
    return resultado