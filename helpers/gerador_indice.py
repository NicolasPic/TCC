import pandas as pd

# ===== Helpers =====
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def trapezio(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if b <= x <= c:      return 1.0
    if a < x < b:        return (x - a) / (b - a)
    if c < x < d:        return (d - x) / (d - c)
    return 0.0

def rise(x, x0, x1):  # 0 em x<=x0, 1 em x>=x1
    if x1 == x0: return 0.0
    return clamp((x - x0) / (x1 - x0))

def fall(x, x0, x1):  # 0 em x>=x0, 1 em x<=x1
    if x0 == x1: return 0.0
    return clamp((x0 - x) / (x0 - x1))

# ===== Parâmetros =====
pesos = {"ph":0.12, "temperature":0.20, "humidity":0.08, "rainfall":0.12, "N":0.16, "P":0.10, "K":0.10}
faixas = {
    "ph": (5.3, 5.8, 6.5, 7.0),
    "temperature": (20, 22, 26, 30),
    "humidity": (45, 55, 70, 80),
    "rainfall": (70, 100, 180, 220),
    "N": (80, 90, 120, 140),
    "P": (15, 20, 35, 45),
    "K": (25, 30, 45, 60),
}

# ===== Índice base =====
def indice_base_row(row):
    s = 0.0
    for v in pesos:
        a, b, c, d = faixas[v]
        s += pesos[v] * trapezio(row[v], a, b, c, d)
    return s

# ===== Penalidades INDIVIDUAIS =====
def pen_ph_p(row):
    # Penalidade por interação entre pH e Fósforo (P):
    # - Quando o pH fica muito alto (>6.8) ou muito baixo (<5.6),
    #   o fósforo disponível no solo diminui drasticamente.
    # - Essa penalidade só é ativada se o fósforo já estiver em nível baixo (P < 25).
    # - Máximo de 10%.
    # severidade de P baixo: 0 se P>=25, 1 se P<=15
    s_P_baixo = fall(row["P"], 25, 15)
    pen_alto  = 0.10 * rise(row["ph"], 6.8, 7.2) * s_P_baixo
    pen_baixo = 0.10 * fall(row["ph"], 5.6, 5.0) * s_P_baixo
    return max(pen_alto, pen_baixo)  # até 10%

def pen_rain_k(row):
    # Penalidade por excesso/falta de chuva em solos com pouco Potássio (K):
    # - Se K < 30, a planta fica mais sensível à variação hídrica.
    # - Chuva alta (>180 mm) pode lavar nutrientes (penaliza até 8%).
    # - Chuva baixa (<90 mm) limita o enchimento do grão (penaliza até 10%).
    # severidade de K baixo: 0 se K>=30, 1 se K<=20
    s_K_baixo = fall(row["K"], 30, 20)
    pen_chuva_alta  = 0.08 * rise(row["rainfall"], 180, 230) * s_K_baixo
    pen_chuva_baixa = 0.10 * fall(row["rainfall"], 90, 60)  * s_K_baixo
    return min(0.10, pen_chuva_alta + pen_chuva_baixa)  # até 10%

def pen_heat_dry(row):
    # Penalidade por estresse de calor + baixa umidade:
    # - Acima de 27 °C a planta entra em estresse térmico.
    # - Se a umidade relativa também estiver baixa (<50%), o estresse se agrava.
    # - Penaliza até 10%.
    return 0.10 * rise(row["temperature"], 27, 32) * fall(row["humidity"], 50, 40)

def pen_doenca(row):
    # Penalidade por risco de doenças fúngicas (ex: ferrugem, cercosporiose):
    # - Ocorrem em condições de alta umidade relativa (>80%) e excesso de chuva (>180 mm).
    # - Penaliza até 5%.
    return 0.05 * rise(row["humidity"], 80, 90) * rise(row["rainfall"], 180, 230)

def pen_nk(row):
    # Penalidade por desbalanço Nitrogênio (N) x Potássio (K):
    # - Excesso de N (>120) sem K suficiente (<30) gera desequilíbrio nutricional.
    # - A planta cresce muito em folha mas com pouca resistência a pragas e seca.
    # - Penaliza até 5%.
    s_K_baixo = fall(row["K"], 30, 20)
    return 0.05 * rise(row["N"], 120, 150) * s_K_baixo

def gerar_indice():
    # ===== Pipeline =====
    df = pd.read_csv(r"C:\TCC\dados\cafe_total.csv")  # ajuste sep=";" se precisar

    # índice base
    df["indice_base"] = df.apply(indice_base_row, axis=1)

    # penalidades individuais
    df["pen_ph_p"] = df.apply(pen_ph_p, axis=1)
    df["pen_rain_k"] = df.apply(pen_rain_k, axis=1)
    df["pen_heat_dry"] = df.apply(pen_heat_dry, axis=1)
    df["pen_doenca"] = df.apply(pen_doenca, axis=1)
    df["pen_nk"] = df.apply(pen_nk, axis=1)

    # teto global 25%
    df["pen_total_bruta"] = df[["pen_ph_p", "pen_rain_k", "pen_heat_dry", "pen_doenca", "pen_nk"]].sum(axis=1)
    df["pen_total"] = df["pen_total_bruta"].clip(upper=0.25)

    # índice final (target para o ML)
    df["indice"] = (df["indice_base"] * (1 - df["pen_total"])).clip(0, 1)

    # ===== Somente colunas para machine learning =====
    cols_ml = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "indice"]
    df_ml = df[cols_ml].copy()

    # arredondar o índice (opcional)
    df_ml["indice"] = df_ml["indice"].round(3)

    # salvar
    df_ml.to_csv(r"C:\TCC\dados\cafe_resultado_final.csv", index=False)

    print(df_ml.head())
