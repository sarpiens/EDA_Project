#Importar librerias de interes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import spiderplot as sp
from itertools import combinations_with_replacement
from scipy.stats import chi2_contingency, kruskal, spearmanr

def min_max_transformation(dataframe:pd.DataFrame,column:str):
    """
    Función para hacer un escalado min-max de una columna de pandas.

    Args:
        dataframe (Pandas DataFrame): El dataframe con el que trabajar.
        column (str): El nombre de la columna del dataframe con la que trabajar.
    
    Returns:
        pandas series: El resultado de aplicar el escalado min-max a la columna.
    """
    # 0. ALgunas validaciones
    if not isinstance(dataframe,pd.DataFrame):
        raise TypeError(f"dataframe debe ser un Pandas DataFrame, pero se proporcionó {type(dataframe)}.")
    if not isinstance(column, str):
        raise TypeError(f"column debe ser un str, pero se proporcionó {type(column)}.")
    # 1. Devolver el resultado del escalado min-max
    return (dataframe[column] - dataframe[column].min())/(dataframe[column].max() - dataframe[column].min())

def get_palette_dict(dataframe:pd.DataFrame, column:str, palette:str, fillNA:str="NA"):
    """
    Genera un dict con colores a partir de la column y sns palette proporcionada.
    
    Args:
        dataframe (Pandas DataFrame): El dataframe con el que trabajar.
        column (str): El nombre de la columna categórica del dataframe con la que trabajar.
        palette (str | dict) : Paleta de colores a utilizar. Proporciona su nombre o un diccionario de grupos:colores.
        fillNA (str) : La cadena por la que sustituir los NaN si es que hay alguno. Default, "NA".
    
    Returns:
        colores_dict (dict) : Diccionario de {grupos:colores}.
    """
    # 0. ALgunas validaciones
    if not isinstance(dataframe,pd.DataFrame):
        raise TypeError(f"dataframe debe ser un Pandas DataFrame, pero se proporcionó {type(dataframe)}.")
    if not isinstance(column, str):
        raise TypeError(f"column debe ser un str, pero se proporcionó {type(column)}.")
    if column not in dataframe.columns:
        raise ValueError("column no es una columna en dataframe.")
    if not isinstance(palette, (str | dict)):
        raise TypeError(f"palette debe ser un str o un dict, pero se proporcionó {type(palette)}.")
    if not isinstance(fillNA, str):
        raise TypeError(f"fillNa debe ser un str, pero se proporcionó {type(fillNA)}.")
    if fillNA in dataframe[column].values:
        raise ValueError(f"{fillNA} ya se encuentra en la column proporcionada, elija otro fillNA para sustituir NAs.")

    #Get list de grupos en column
    column_groups = list(dataframe[column].fillna(fillNA).unique()) #Sustituimos NaNs por fillNA
    #Get sns_palette
    sns_palette = sns.color_palette(palette, len(column_groups))
    #Generar dict de colores
    colores_dict = dict(zip(column_groups,sns_palette))
    
    #Devolver dict
    return colores_dict

def plot_categorica_seaborn(dataframe:pd.DataFrame, column:str, palette:str | dict ="viridis", plot_size:tuple=(14,5), show_values:bool=True, xlabel45rotation:bool=True, original_order:bool=False, fillNA:str="NA"):
    """
    Para una variable categorica genera una figura de Matplotlib/Seaborn con gráficos de conteo y frecuencia relativa(%).
    
    Args:
        dataframe (Pandas DataFrame): El dataframe con el que trabajar.
        column (str): El nombre de la columna categórica del dataframe con la que trabajar.
        palette (str | dict) : Paleta de colores a utilizar. Proporciona su nombre o un diccionario de grupos:colores.
        plot_size (tuple) : El tamaño de la figura final (anchura, altura).
        show_values (bool) : Indica si muestra los valores encima de las barras. Default, True.
        xlabel45rotation (bool) : Indica si debe rotar 45 grados las etiquetas del eje X. Default, True.
        original_order (bool): Indica si se debe mantener el orden original de la columna. Default, False.
        fillNA (str) : La cadena por la que sustituir los NaN si es que hay alguno. Default, "NA".

    Returns:
        fig (matplotlib.figure.Figure) : Devuelve la figura con gráficos de conteo y frecuencia relativa(%).
    """
    # 0. ALgunas validaciones
    if not isinstance(dataframe,pd.DataFrame):
        raise TypeError(f"dataframe debe ser un Pandas DataFrame, pero se proporcionó {type(dataframe)}.")
    if not isinstance(column, str):
        raise TypeError(f"column debe ser un str, pero se proporcionó {type(column)}.")
    if column not in dataframe.columns:
        raise ValueError("column no es una columna en dataframe.")
    if not isinstance(palette, (str | dict)):
        raise TypeError(f"palette debe ser un str o un dict, pero se proporcionó {type(palette)}.")
    if not isinstance(plot_size, (tuple)):
        raise TypeError(f"plot_size debe ser una tuple, pero se proporcionó {type(plot_size)}.")
    if len(plot_size) != 2:
        raise ValueError(f"plot_size debe tener tamaño 2 (anchura, altura).")
    if not all(isinstance(x, int | float) and x > 0 and not isinstance(x, bool) for x in plot_size):
        raise TypeError(f"Todos los valores dentro de plot_size deben ser números positivos mayores que cero.")
    if not isinstance(show_values,bool):
        raise TypeError(f"show_values debe ser un bool, pero se proporcionó {type(show_values)}.")
    if not isinstance(xlabel45rotation,bool):
        raise TypeError(f"xlabel45rotation debe ser un bool, pero se proporcionó {type(xlabel45rotation)}.")
    if not isinstance(original_order,bool):
            raise TypeError(f"original_order debe ser un bool, pero se proporcionó {type(original_order)}.")
    if not isinstance(fillNA, str):
        raise TypeError(f"fillNa debe ser un str, pero se proporcionó {type(fillNA)}.")
    if fillNA in dataframe[column].values:
        raise ValueError(f"{fillNA} ya se encuentra en la column proporcionada, elija otro fillNA para sustituir NAs.")
    
    # 1. Preparaciones iniciales
    ## 1.1. Preparamos la palette
    num_groups = dataframe[column].nunique(dropna=False) #Incluimos NAs
    if type(palette) == str:
        validated_palette = sns.color_palette(palette, n_colors=num_groups)
    if type(palette) == dict:
        validated_palette = palette
    
    ##1.2. Generar tablas con pandas
    ##Nota: Al usar value_counts también ordenamos las barras(value_counts: sort=True, por defecto)
    df_counts = dataframe[column].value_counts(dropna=False).reset_index().fillna(fillNA) #Incluimos NAs y los sustituimos por "fillNA"
    df_percentages = (dataframe[column].value_counts(dropna=False, normalize=True)* 100).reset_index().fillna(fillNA) #Incluimos NAs y los sustituimos por "fillNA"
    df_percentages.rename(columns={'proportion': 'percentage'}, inplace=True) #cambiamos nombre columna

    # 2. Generamos la figura
    ## 2.1. Inicializar la figura con dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=plot_size, layout="constrained")
    sns.set_theme(style="whitegrid") #Set theme style

    ## 2.2. Definir orden segun original_order parameter
    if original_order==True:
        bars_order = list(dataframe[column].fillna(fillNA).unique()) #Sustituimos NaNs por fillNa
    else:
        bars_order = None

    ## 2.3. GRÁFICO 1: Conteos Absolutos
    sns.barplot(
        x=column,
        y="count",
        data=df_counts,
        hue=column, #Le indicamos que use column para rellenar colores
        palette=validated_palette,
        order=bars_order,
        legend=False,
        dodge=False, #Para que centre los barplots
        ax=axes[0], #Guardamos en figura (primer slot)
    )
    ## 2.4. GRÁFICO 2: Porcentajes (%)
    sns.barplot(
        x=column,
        y="percentage",
        data=df_percentages,
        hue=column, #Le indicamos que use column para rellenar colores
        palette=validated_palette,
        order=bars_order,
        legend=False,
        dodge=False, #Para que centre los barplots
        ax=axes[1], #Guardamos en figura (segundo slot)
    )
    ## 2.5. Set labels eje x e y
    axes[0].set(ylabel="Absolute Counts",xlabel=None)
    axes[1].set(ylabel="Percentage (%)",xlabel=None)
    #fig.suptitle(f"Frequency Plots for {column} ({len(dataframe)} observations)", fontsize=16, fontweight='bold')
    ## 2.6. Mostrar valores encima de las barras en funcion de show_values
    if show_values==True:
        #Mostrar conteos del primer gráfico
        for contenedor0 in axes[0].containers: axes[0].bar_label(contenedor0,padding=3)
        #Mostrar porcentages 2 primeras cifras decimales
        for contenedor1 in axes[1].containers: axes[1].bar_label(contenedor1,fmt="%.2f", padding=3)
    ##2.7. Indicar si giramos labels eje x 45 grados
    if xlabel45rotation==True:
        for ax in fig.axes: ax.tick_params(axis='x', labelrotation=45)
    ## 2.8. Cerramos la figura
    plt.close(fig)
    # 3. Return final plot
    return fig

def plot_boxplot_custom(dataframe:pd.DataFrame, column:str, plot_size:tuple=(15,5), boxplot_whis=1.5, vertical:bool=True):
    """
    Genera un boxplot custom junto con su media y std (error bars).

    Args:
        dataframe (Pandas DataFrame): El dataframe con el que trabajar.
        column (str): El nombre de la columna numérica del dataframe con la que trabajar.
        plot_size (tuple) : El tamaño de la figura final (anchura, altura).
        boxplot_whis (int | float) : Indicar el valor usado para whiskers k * IQR. Default, 1.5.
        vertical (bool) : Indica si el boxplot se genera con orientación vertical. Default, True.as_integer_ratio
    
    Returns:
        fig (matplotlib.figure.Figure) : Devuelve el boxplot final.
    """
    # 0. ALgunas validaciones
    if not isinstance(dataframe,pd.DataFrame):
        raise TypeError(f"dataframe debe ser un Pandas DataFrame, pero se proporcionó {type(dataframe)}.")
    if not isinstance(column, str):
        raise TypeError(f"column debe ser un str, pero se proporcionó {type(column)}.")
    if column not in dataframe.columns:
        raise ValueError("column no es una columna en dataframe.")
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise TypeError(f"column debe tener un dtype numérico, pero presenta dtype {dataframe[column].dtype}.")
    if not isinstance(plot_size, (tuple)):
        raise TypeError(f"plot_size debe ser una tuple, pero se proporcionó {type(plot_size)}.")
    if len(plot_size) != 2:
        raise ValueError(f"plot_size debe tener tamaño 2 (anchura, altura).")
    if not all(isinstance(x, int | float) and x > 0 and not isinstance(x, bool) for x in plot_size):
        raise TypeError(f"Todos los valores dentro de plot_size deben ser números positivos mayores que cero.")
    if not isinstance(boxplot_whis,int | float) or isinstance(boxplot_whis, bool):
        raise TypeError(f"boxplot_whis debe ser un número, pero se proporcionó {type(boxplot_whis)}.")
    if boxplot_whis <= 0:
        raise ValueError("boxplot_whis debe ser un número mayor que cero.")
    if not isinstance(vertical,bool):
        raise TypeError(f"vertical debe ser un bool, pero se proporcionó {type(vertical)}.")

    # 1. Definimos x and y values based on vertical
    if vertical == True:
        x_value = column
        y_value = None
    else:
        x_value = None
        y_value = column
    
    #2. Generamos el gráfico boxplot
    ## 2.1.Definimos tamaño
    #plt.figure(figsize=plot_size)
    fig, ax = plt.subplots(figsize=plot_size)
    ## 2.2. Boxplot inicial
    sns.boxplot(data=dataframe,x=x_value,y=y_value,fliersize=5, linewidth=2, whis=boxplot_whis, zorder=2, ax=ax)
    
    ## 2.3. Añadimos un pointplot con errorbars para mostrar media y sd
    sns.pointplot(data=dataframe, x=x_value, y=y_value, estimator="mean",  
                  markerfacecolor="yellow", markeredgecolor="black", markers="s", zorder=3,
                  errorbar="sd",capsize=0.2, err_kws={"color": "red","linewidth": 4,"zorder": 1}, # zorder, ENVÍA LA BARRA DETRÁS (Capa inferior al triángulo)
                  ax= ax #Le indicamos que dibuje en el previo
                  )  
    ## 2.4. Cerramos la figura
    plt.close(fig)
    # 3. Return final plot
    return fig

def plot_multiple_custom_boxplots(dataframe:pd.DataFrame, columns:list, n_cols:int, n_rows:int, plot_size:tuple=(20,5), boxplot_whis=1.5):
    """
    Genera multiples custom boxplos en una única figura usando una lista de columnas numericas de interés.

    Args:
        dataframe (Pandas DataFrame): El dataframe con el que trabajar.
        columns (list): Lista con columnas numéricas de interés.
        n_cols (int) : Número de columnas.
        n_rows (int) : numero de filas. 
        plot_size (tuple) : El tamaño de la figura final (anchura, altura).
        boxplot_whis (int | float) : Indicar el valor usado para whiskers k * IQR. Default, 1.5.
    
    Returns:
        fig (matplotlib.figure.Figure) : Devuelve lq figura final con totos los boxplots.
    """
    #1. empieza el plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_size)
    axes = axes.flatten()

    #2. Generamos boxplots = a plot_boxplot_custom para cada columna
    for i, column in enumerate(columns):
        if dataframe[column].dtype in ['int64', 'float64']:
            #Generamos boxplot
            sns.boxplot(data=dataframe, y=column, fliersize=5, linewidth=2, whis=boxplot_whis, zorder=2, ax=axes[i])
            #Añadimos media y std
            sns.pointplot(data=dataframe, y=column, estimator="mean",  
                              markerfacecolor="yellow", markeredgecolor="black", markers="s", zorder=3,
                              errorbar="sd",capsize=0.2, err_kws={"color": "red","linewidth": 4,"zorder": 1}, # zorder, ENVÍA LA BARRA DETRÁS (Capa inferior al triángulo)
                              ax= axes[i] #Le indicamos que dibuje en el previo
                              )
    #3. Para añadir espacio entre plots
    plt.subplots_adjust(wspace=0.5)
    #4. Cerramos la figura
    plt.close(fig)
    #5. Return final plot
    return fig

def plot_multiple_hist(dataframe:pd.DataFrame, columns:list, n_cols:int, n_rows:int, plot_size:tuple=(20,5), bins="auto"):
    """
    Genera multiples histogramas en una única figura usando una lista de columnas numericas de interés.

    Args:
        dataframe (Pandas DataFrame): El dataframe con el que trabajar.
        columns (list): Lista con columnas numéricas de interés.
        n_cols (int) : Número de columnas.
        n_rows (int) : numero de filas. 
        plot_size (tuple) : El tamaño de la figura final (anchura, altura).
        boxplot_whis (int | float) : Indicar el valor usado para whiskers k * IQR. Default, 1.5.
    
    Returns:
        fig (matplotlib.figure.Figure) : Devuelve la figura final con todos los histogramas.
    """
    #1. empieza el plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_size)
    axes = axes.flatten()

    #2. Generamos boxplots = a plot_boxplot_custom para cada columna
    for i, column in enumerate(columns):
        if dataframe[column].dtype in ['int64', 'float64']:
            #Generamos hist con KDE
            sns.histplot(dataframe[column], kde=True, ax=axes[i], bins= bins)
        
    #3. Para añadir espacio entre plots
    plt.subplots_adjust(wspace=0.5)
    #4. Cerramos la figura
    plt.close(fig)
    #5. Return final plot
    return fig

def chi2_contingency_results(contigency_table):
    """
    Fúnción para realizar chi2. Muestras los resultados por pantalla y los devuelve como un tabla.

    Args:
        contigency_table
    
    Returns:
        chi2test results
    """
    #Proporcionamos la tabla de contingencia
    chi2_test = chi2_contingency(contigency_table)
    #Mostrar resultados
    print("Resultados:")
    print("- Valor Chi-Cuadrado:", chi2_test[0])
    print("- P-Value:", chi2_test[1])
    print("- Grados de Libertad:", chi2_test[2])
    #Comprobamos condiciones de aplicabilidad
    print("Condiciones de aplicabilidad:")
    ##Calculos
    total_celdas = chi2_test[3].size
    celdas_valor_5 = np.sum(chi2_test[3] >= 5)
    porcentaje_valor_5 = (celdas_valor_5 / total_celdas) * 100
    tiene_menor_1 = np.any(chi2_test[3] < 1)
    #Resultados
    print(f"- ¿Hay alguna celda esperada < 1?: {'Sí' if tiene_menor_1 else 'No'}")
    print(f"- Porcentaje de celdas esperadas >= 5: {porcentaje_valor_5:.2f}%")
    #Mensaje interpretacion
    if tiene_menor_1 or porcentaje_valor_5 < 80:
        print("ADVERTENCIA: No se cumplen las condiciones de aplicabilidad.")
        print("El p-valor obtenido mediante Chi-cuadrado puede no ser confiable.")
    else:
        print("¡ÉXITO! Se cumplen las condiciones de aplicabilidad.")
    #devolver el resultado
    return chi2_test

def facet_spider_plot(dataframe, other_categories:list, palette, n_cols:int, n_rows:int,
                      stat_col="stat", value_col="value", category_col="category", global_category="Global", 
                      plot_size:tuple=(20,5), radar_limits=[-0.1,0.45]):
    """
    Genera varios spider plot y devuelve la figura.
    """
    #Usaremos la libreria spyderplot(sp)
    ##Set style
    sns.set_style("whitegrid")

    #1. empieza el plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_size, subplot_kw={'projection': 'polar'})
    axes_planos = axes.flatten()

    #2. Generamos plot para cada categoria other
    for i,cat in enumerate(other_categories):
        #Filtramos la tabla para tener stats globales y de la categoria actual(tmp_categ)
        tmp_filtered_df = dataframe[dataframe[category_col].isin([cat,global_category])]
        #Seleccionamos espacio en la cuadricula
        ax_espacio = axes_planos[i]
        #ax_espacio.set_aspect('equal', adjustable='box')
        #Generamos el radar
        ax_radar = sp.spiderplot(data=tmp_filtered_df, x=stat_col, y=value_col, hue=category_col, style=category_col,
                      dashes=False, palette=palette, ax=ax_espacio)
        #Definimos limites
        ax_radar.set_title(cat, fontweight='bold', fontsize=10) 
        ax_radar.set_rlim(radar_limits)
        #Movemos las etiquetas stat fuera del gráfico y cambiamos color a negro
        ax_radar.tick_params(axis="x", pad=15, colors="black", labelsize=7)
        #Movemos las etiqyetas de y=value a grado 22.5
        ax_radar.set_rlabel_position(22.5)
        #Movemos ambos grupos etiquetas al frente
        ax_radar.yaxis.set_zorder(10)
        ax_radar.xaxis.set_zorder(11)
        #Ola leyenda
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.5, 1.3), borderaxespad=0.)

    # Ocultamos los subplots polares que queden vacíos si hay menos categorías que espacios en la cuadrícula
    for j in range(len(other_categories), len(axes_planos)):
        fig.delaxes(axes_planos[j])
    #3. Para añadir espacio entre plots
    fig.tight_layout(pad=2.0, w_pad=0.5, h_pad=0.5)
    #plt.subplots_adjust(hspace=0.3, wspace=0.3)
    #4. Cerramos la figura
    plt.close(fig)
    #5. Return final plot
    return fig

def facet_violinplots_by_category(dataframe,cat_column,num_columns,order_cats,palette_cats,n_cols=3,n_rows=3, plot_size=(14,10)):
    """
    Genera varios violinplots y devuelve la figura.
    """
    #1. empieza el plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_size)
    axes = axes.flatten()
    
    #2. Generamos boxplots = a plot_boxplot_custom para cada columna
    for i, column in enumerate(num_columns):
        if dataframe[column].dtype in ['int64', 'float64']:
            #Generamos boxplot
            sns.violinplot(data=dataframe, y=column,x=cat_column, hue=cat_column, legend=False, 
                           cut=0, #Para que no salgan negativos en las colas; Corta de forma estricta en el valor mínimo y máximo absoluto de tu dataset. 
                           order=order_cats, palette=palette_cats, ax=axes[i])
            #Añadimos media
            sns.pointplot(data=dataframe, y=column, x=cat_column, estimator="mean", errorbar=None,  
                          markerfacecolor="yellow", markeredgecolor="black", markers="s", ax= axes[i])
            #Ajustes del plot            
            axes[i].set_xlabel(None)
    #3. Para añadir espacio entre plots
    plt.subplots_adjust(wspace=0.5)
    for ax in fig.axes: ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout(pad=2.0, w_pad=0.5, h_pad=0.5)
    #4. Cerramos la figura
    plt.close(fig)
    #5. Return final plot
    return fig

def matriz_kruskal_wallis(dataframe, variables_categoricas, variables_numericas, alpha=0.05):
    """
    Genera una tabla resumen con los resultados del test de Kruskal-Wallis para todas las combinaciones posibles de variables categóricas y numéricas.
    """
    # Iniciar lista resultados
    resultados = []

    # Iterar por cada par categórico-numérico
    for var_cat in variables_categoricas:
        for var_num in variables_numericas:

            # 1. Agrupar los datos numéricos por cada categoría única (limpiando nulos)
            # Agrupamos solo si la categoría tiene datos válidos
            grupos = [
                sub_df[var_num].dropna()
                for nombre, sub_df in dataframe.groupby(var_cat)
            ]

            # 2. Calcular el test de Kruskal-Wallis
            stat_h, p_valor = kruskal(*grupos)

            # 3. Determinar si hay diferencias significativas (alfa = 0.05)
            es_significativo = "Sí" if p_valor <= alpha else "No"

            # 4. Guardar los datos en un diccionario temporal
            resultados.append(
                    {
                        "Variable_Categorica": var_cat,
                        "Variable_Numerica": var_num,
                        "Test": "Kruskal-Wallis",
                        "Estadistico_H": round(stat_h, 4),
                        "P_Valor": p_valor,
                        "Alpha": alpha,
                        "Significativo": es_significativo,
                    }
                )
    # 5. Convertir la lista de resultados en un DataFrame y devolver
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

def matriz_spearman(dataframe_nums, alpha=0.05):
    """
    Genera un dataframe de resultados del test de correlación de Spearman.
    """
    # 1. Creamos una lista vacía para almacenar los diccionarios de cada par
    lista_resultados = []

    # 2. Iteramos de forma eficiente sobre combinaciones únicas, incluyendo las self de la diagonal (evita duplicados)
    for col1, col2 in combinations_with_replacement(dataframe_nums.columns, 2):
        # Combinamos.dropna() asegura que los valores nulos no rompan el cálculo de SciPy
        datos_limpios = dataframe_nums[[col1, col2]].dropna()

        # Calculamos coeficiente (rho) y p-valor con SciPy
        stat, p_val = spearmanr(datos_limpios.iloc[:,0], datos_limpios.iloc[:,1])

        # Guardamos los datos estructurados
        lista_resultados.append(
            {
                'Variable 1': col1,
                'Variable 2': col2,
                'Test':"Spearman",
                'Estadístico (rho)': round(stat, 4),
                'p-valor': round(p_val, 5),
                'Alpha':alpha,
                'Significativo': 'Sí' if p_val <= 0.05 else 'No',
            }
    )
    # 4. Convertimos la lista final en el DataFrame y lo devolvemos
    df_reporte_corr = pd.DataFrame(lista_resultados)
    return df_reporte_corr