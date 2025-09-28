from helpers.gerador_indice import gerar_indice
from machines.randon_florest import treinar_modelo, usar_modelo


def main():
    #gerar_indice()
    #treinar_modelo(r"C:\TCC\dados\cafe_resultado_final_redondo.csv", r"C:\TCC\machines\modelos\randon_florest_modelo.joblib")

    usar_modelo(
        N=100,  # dentro do ideal 90–120
        P=25,  # dentro do ideal 20–35
        K=38,  # dentro do ideal 30–45
        temperature=24,  # dentro do ideal 22–26
        humidity=60,  # dentro do ideal 55–70
        ph=6.2,  # dentro do ideal 5.8–6.5
        rainfall=150,  # dentro do ideal 100–180
        multiplicar_por_60=True
    )
if __name__ == "__main__":
    main()