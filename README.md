#Modelo da séries temporais - ARIMA

 **ARIMA:** Modelos autorregressivos integrados e de médias móveis 

 **Objetivo:** Implementação do modelo autorregressivo  de series temporais diposnível na biblioteca __statsmodels.tsa.arima.model__,  a fim de se determinar precipitação chuvosa.

  ## Modelos Uitlizados
 **Média móvel (MA):** Indica que o erro de regressão é uma combinação linear dos termos de erro dos valores passados.

 
 **Parâmetros ARIMA (p, d, q):**
     - p = ordem da autorregressão.
    - d = grau de diferenciação.
    - q = ordem da média móvel.(Combinação linear de erros passados)
    
 **Parâmetros Media móvel (0,0,q):**


### Série temporal estudada
1. Série temporal
![serie_temporal_estudada](serie_temporal_estudada.png )
1.1 Média movel sete dias serie temporal.
![media_movel_serie_temporal](media_movel_serie_temporal.png )
1.2 Série tempora decomposta.
![serie_temporal_decomposta](serie_temporal_decomposta.png )
1.3  Autocorrelation Function -Série Original
![autocorrelacao_raiz_cubica](diagrama_acf_serie_real.png) 
1.4 Partial Autocorrelation Function-Série Original
![parcial_autocorrelacao](diagrama_pacf_serie_real.png )



2. Transformação raíz cubica sobre a série
2.1 Série transformada
![serie_raiz_cubica](serie_raiz_cubica.png )
2.2  Quantile-Quantile Plot -Série raíz cúbica
![nomral_qq_plt_serie_raiz_cubica](teste_normalidade_raiz_cubica.png) 
2.3  Autocorrelation Function -Série raíz cúbica
![autocorrelacao_raiz_cubica](diagrama_acf_serie_cubica.png) 
2.4 Partial Autocorrelation Function-Série raíz cúbica
![parcial_autocorrelacao](diagrama_pacf_serie_cubica.png)


3. Aplicação Modelo Media Movel 
3.1 Estudo dos resíduos
![residuos_modelo_media_movel](residuos_modelo_media_movel.png)

3.2 Quantile-Quantile Plot-Rediduo -Modelo Média Movel
![Quantile-Quantile Plot-Rediduo Série raíz cúbica](teste_normalidade_residuos_ma.png) 

3.3 Autocorrelation Function-Resíduos -Modelo MA.
![Autocorrelation Function -Resíduos- -Modelo MA](diagrama_acf_ma.png)

3.4. Partial Autocorrelation Function -Resíduos- -Modelo MA.
![Parcial autocorreção residuos](diagrama_pacf_ma.png)

4 Resultados

4.1 Série e resíduos
![seire_e_residuos](serie_e_residuo.png)

4.2 Previsão e resíduo
![seire_e_residuos](previsao_e_residuo.png)

