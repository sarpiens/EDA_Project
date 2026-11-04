# Proyecto EDA

## Índice

- [**Descripción**](#descripción)
- [**Contexto**](#contexto)
- [**Archivos**](#archivos)
- [**Tecnologías y Prerrequisitos**](#tecnologías-y-prerrequisitos)
- [**Instalación**](#instalación)

## Descripción

Proyecto individual EDA para el Curso de Data Science - The Bridge y Fundación Universia (2026-09). En este proyecto hemos realizado un análisis exploratorio de datos (EDA).

## Contexto

En este caso, he decidido centrar la temática en un EDA dirigido a investigar las características de los Pokémon, como un ejemplo educativo y de interés para los amantes de este videojuego clásico. Para simplificar, he optado por realizar el análisis centrándome en los Pokémon en su forma base (aquellos en su primera evolución o que poseen una única forma evolutiva).

Como fuente de datos se ha utilizado **PokéAPI** (https://pokeapi.co/), una API gratuita y de uso libre con fines educativos. Según se indica en su **repositorio de GitHub** (https://github.com/pokeapi/pokeapi), el proyecto PokéAPI está sujeto a la licencia BSD-3-Clause. Por otro lado, cabe aclarar que Pokémon y sus personajes son propiedad registrada de Nintendo, Game Freak y The Pokémon Company. Este análisis se desarrolla sin fines comerciales operando bajo el principio de Fair Use (uso legítimo no comercial).

## Archivos

* El código utilizado para la obtención de la información de PokéAPI, se puede consultar en el notebook **1.PokeAPI.ipynb**.

* El código utilizado para realizar el proceso de limpieza, derivación de variables y preparación del dataframe final usado para los análisis, se puede consultar en el notebook **2.Preparacion_dataframe.ipynb**.

* El código utilizado para realizar los análisis y la memoria, se pueden consultar en el notebook **3.Memoria.ipynb**.

* El código con las funciones desarrolladas para la preparación y los análisis, se puede consultar en el script **utils.py**.

* La presentación puede consultarse en el archivo **Proyecto2EDA.pdf**.

## Tecnologías y Prerrequisitos

- **Lenguaje:** Python. Este análisis está desarrollado en **Python 3.12.10**. Los notebooks y scripts utilizan varias librerías integradas en la biblioteca estándar de Python, por lo que **no requieren instalación adicional**:
   * `time`
   * `itertools`

- **Librerías Externas:** 
Para que el análisis funcione correctamente, es necesario instalar las siguientes librerías de terceros (se incluye un archivo requirements.txt indicando las versiones de las librerías):
   * `numpy`
   * `pandas`
   * `matplotlib`
   * `seaborn`
   * `spiderplot`
   * `requests`
   * `scipy`

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/sarpiens/EDA_Project.git
```
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```
3. Explora los notebooks y la presentación.
