from cProfile import label
from re import L
from tkinter import Label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn import datasets
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
from tratamento_dataset import tratamento_dataset
from testa_hipotese_shapiro_wilk import testa_hipotese_shapiro_wilk 
from estudo_media_movel import estudo_media_movel
"""

from estudo_modelo_ar import estudo_modelo_ar
from testa_hipotese_kpss import testa_hipotese_kpss
from testa_hipotese_shapiro_wilk import testa_hipotese_shapiro_wilk"""



rcParams['figure.figsize'] =[15,6]
alpha =0.05 # Significancia 

#Tratamento de dados faltantes e tipos
data_set_bruto =pd.read_csv('Chuva_Mensal.csv', sep=';')
print(f'{data_set_bruto.head()}')
tratamento_dataset(data_set_bruto)
del data_set_bruto

#--------------------Configuração base de dados--------------------
data_set =pd.read_csv('data_set_tratado.csv')
data_set =data_set.drop(columns='Ano')
data_set_array =data_set.values

data_set_list =list(data_set_array.flatten())
indice =pd.date_range('1985', periods=len(data_set_list), freq='M')

serie_tempora_chuva =pd.Series(data_set_list,index=indice )

serie_tempora_chuva.plot(label ='Preciptação chuvosa')
plt.xlabel('Ano')
plt.ylabel('Preciptação de chuvosa em mm')
plt.title("Preciptação chovosa ao longo dos anos")
plt.legend(loc ='best')
plt.savefig('serie_temporal_estudada.png', dpi =300, format ='png')
plt.show()

plot_acf(serie_tempora_chuva, lags= 30)
plt.title("Diagrama Autocorrelação Série Real")
plt.xlabel('Nº Lags')
plt.ylabel('Autocorrelação')
plt.savefig('diagrama_acf_serie_real.png', dpi =300, format ='png')
plt.show()

plot_pacf(serie_tempora_chuva, lags= 30)
plt.title("Diagrama Autocorrelação Parcial Real")
plt.xlabel('Nº Lags')
plt.ylabel('Autocorrelação parcial ')
plt.savefig('diagrama_pacf_serie_real.png', dpi =300, format ='png')
plt.show()

estudo_media_movel(serie_tempora_chuva) 


#Decomposição da serie, a fim de analisar tendência, sazonalidade e resíduos.

serie_tempora_chuva_raiz_cubica = serie_tempora_chuva**(1/3)




serie_decomposta = seasonal_decompose(serie_tempora_chuva,model = "additive" )
serie_decomposta.plot()
plt.savefig('serie_temporal_decomposta.png', dpi =300, format ='png')
plt.show()
#--------------------------------------------------------------------

#-----------------Traformação raiz cubica sobre a serie--------------


serie_tempora_chuva_raiz_cubica.plot(label ="Serie raíz cubica")
plt.title("Série tranformação raíz cúbica")
plt.legend(loc ='best')
plt.xlabel('Anos')
plt.ylabel('Raíz cúbica da série')
plt.savefig('serie_raiz_cubica.png', dpi =300, format ='png')
plt.show()

# Teste de normalidade
stats.probplot(serie_tempora_chuva_raiz_cubica, dist ="norm", plot =plt )
plt.savefig('teste_normalidade_raiz_cubica.png', dpi =300, format ='png')
plt.show()

# Diagrama ACF e PACF 
plot_acf(serie_tempora_chuva_raiz_cubica, lags= 30)
plt.title("Diagrama Autocorrelação Série Raíz Cúbica")
plt.xlabel('Nº Lags')
plt.ylabel('Autocorrelação')
plt.savefig('diagrama_acf_serie_cubica.png', dpi =300, format ='png')
plt.show()

plot_pacf(serie_tempora_chuva_raiz_cubica, lags= 30)
plt.title("Diagrama Autocorrelação Parcial Série Raíz Cúbica")
plt.xlabel('Nº Lags')
plt.ylabel('Autocorrelação parcial ')
plt.savefig('diagrama_pacf_serie_cubica.png', dpi =300, format ='png')
plt.show()


modelo_ma =ARIMA(serie_tempora_chuva_raiz_cubica, order =(0,0,9))
resultado_ma =modelo_ma.fit()
print(f'{resultado_ma.summary()}')

residuo_ma =resultado_ma.resid

residuo_ma.plot(label ='Resíduos')
plt.title("Residuos modelo media movel")
plt.xlabel('Anos')
plt.legend(loc ='best')
plt.ylabel('Residuos')
plt.savefig('residuos_modelo_media_movel.png', dpi =300, format ='png')
plt.show()


# Teste de normalidade
stats.probplot(residuo_ma, dist ="norm", plot =plt )
plt.savefig('teste_normalidade_residuos_ma.png', dpi =300, format ='png')


#Teste de Shapiro-Wilk 
statistic,value_p =stats.shapiro(residuo_ma)
print('Métricas  Shapiro-Wilk:\n')
print('Estatistica do teste: {:.6f} '.format(statistic))
print('p-valor: {:.6f} '.format(value_p))
testa_hipotese_shapiro_wilk(value_p, alpha)
#-----------------------------------------------------------------

sns.displot(residuo_ma, label ='Resíduo')
plt.title("Histograma Residuo")
plt.xlabel('Residuo')
plt.legend(loc ='best')
plt.ylabel('Densidade')
plt.savefig('histograma_residuo_ma.png', dpi =300, format ='png')
plt.show()



plot_acf(residuo_ma, lags= 30)
plt.title("Diagrama Autocorrelação Resíduos")
plt.xlabel('Nº Lags')
plt.ylabel('Autocorrelação')
plt.savefig('diagrama_acf_ma.png', dpi =300, format ='png')
plt.show()

# Diagrama ACF e PACF 

plot_pacf(residuo_ma, lags= 30)
plt.title("Diagrama Autocorrelação Parcial Resíduos")
plt.xlabel('Nº Lags')
plt.ylabel('Autocorrelação parcial ')
plt.savefig('diagrama_pacf_ma.png', dpi =300, format ='png')
plt.show()



#Visualização
plt.plot(serie_tempora_chuva_raiz_cubica, color ='blue', label='Série real')
plt.plot(serie_tempora_chuva_raiz_cubica-residuo_ma,color ='green', label='Resíduos')
plt.legend(loc ='best')
plt.title('Comparativo Série e Resíduos')
plt.xlabel('Anos')
plt.savefig('serie_e_residuo.png', dpi =300, format ='png')
plt.show()

#----------------------------------------------------------------------------

# Modelos da série
print(f'{resultado_ma.fittedvalues}')  
previsao_ma =resultado_ma.predict(start =431, end =443)

#Previsao

print(f'{previsao_ma} ')
plt.plot(serie_tempora_chuva_raiz_cubica, color ='blue', label='Série real')
plt.plot(previsao_ma, color ='green', label='Previsto')
plt.plot(serie_tempora_chuva_raiz_cubica-residuo_ma, color ='black', label='Resíduos')
plt.legend(loc ='best')
plt.xlabel('Anos')
plt.savefig('previsao_e_residuo.png', dpi =300, format ='png')
plt.show()

