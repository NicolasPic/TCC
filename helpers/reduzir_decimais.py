import pandas as pd

def reduzir_decimais_csv(input_path, output_path, casas=3):
    """
    Lê um CSV, arredonda todas as colunas numéricas para 'casas' decimais
    e salva em outro CSV.
    """
    # lê o csv
    df = pd.read_csv(input_path)

    # identifica colunas numéricas
    num_cols = df.select_dtypes(include=["float", "int"]).columns

    # arredonda
    df[num_cols] = df[num_cols].round(casas)

    # salva
    df.to_csv(output_path, index=False)

    print(f"Arquivo salvo em: {output_path}")
    print(df.head())