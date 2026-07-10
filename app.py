import json
import unicodedata

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Analítica Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)


def limpiar_nombre(texto):
    if pd.isna(texto):
        return ""

    texto = str(texto)
    texto = texto.replace("Provincia de ", "")
    texto = texto.replace(" Province", "")
    texto = texto.replace("(38)", "")
    texto = texto.replace("(23)", "")
    texto = texto.replace("Comarca Guna Yala", "Comarca Kuna Yala")
    texto = texto.replace("Comarca Emberá-Wounaan", "Comarca Emberá")
    texto = texto.replace("Comarca Ngäbe-Buglé", "Comarca Ngäbe Buglé")
    texto = texto.strip()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    return texto.lower()


@st.cache_data
def cargar_datos_ames():
    try:
        return pd.read_csv("AmesHousing_limpio.csv")
    except FileNotFoundError:
        return pd.read_csv("AmesHousing.csv")


@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load("mejor_modelo_regresion.pkl")
    except Exception:
        return None


@st.cache_data
def cargar_datos_panama():
    df_panama = pd.read_csv(
        "panama_division_mapa.csv",
        sep=";"
    )

    df_panama = df_panama[
        (df_panama["nivel"] == "provincia")
        & (df_panama["provincia"] != "TOTAL")
    ].copy()

    df_panama["provincia_limpia"] = (
        df_panama["provincia"].apply(limpiar_nombre)
    )

    df_panama["viviendas_total"] = pd.to_numeric(
        df_panama["viviendas_total"],
        errors="coerce"
    )

    df_panama = df_panama.dropna(
        subset=["provincia_limpia", "viviendas_total"]
    )

    with open(
        "geoBoundaries-PAN-ADM1_simplified.geojson",
        encoding="utf-8"
    ) as archivo_geojson:
        geojson_panama = json.load(archivo_geojson)

    for feature in geojson_panama["features"]:
        shape_name = feature["properties"].get("shapeName", "")

        feature["properties"]["provincia_limpia"] = limpiar_nombre(
            shape_name
        )

    return df_panama, geojson_panama


df = cargar_datos_ames()
modelo_reg = cargar_modelo()
df_panama, geojson_panama = cargar_datos_panama()


variables_modelo = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Garage Area",
    "Total Bsmt SF",
    "Year Built",
    "Full Bath",
    "Bedroom AbvGr"
]


st.title("Sistema Integrado de Analítica Inmobiliaria")

st.write(
    "Proyecto Final - Ames Housing y "
    "Mapa Sociodemográfico de Panamá"
)


tab1, tab2, tab3 = st.tabs(
    [
        "📈 Análisis Exploratorio",
        "🗺️ Mapa de Panamá",
        "🔮 Simulador Predictivo"
    ]
)


with tab1:
    st.header(
        "Análisis Exploratorio del Dataset Ames Housing"
    )

    vecindarios = sorted(
        df["Neighborhood"].dropna().unique().tolist()
    )

    seleccion_vecindarios = st.multiselect(
        "Seleccione los vecindarios:",
        options=vecindarios,
        default=vecindarios[:4]
    )

    if not seleccion_vecindarios:
        seleccion_vecindarios = vecindarios

    dff = df[
        df["Neighborhood"].isin(seleccion_vecindarios)
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cantidad de viviendas",
        f"{len(dff):,}"
    )

    col2.metric(
        "Precio promedio",
        f"${dff['SalePrice'].mean():,.2f}"
    )

    col3.metric(
        "Precio máximo",
        f"${dff['SalePrice'].max():,.2f}"
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            dff,
            x="SalePrice",
            title="Distribución de precios de venta"
        )

        fig_hist.update_layout(
            xaxis_title="Precio de venta",
            yaxis_title="Cantidad de viviendas"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    with c2:
        promedios = (
            dff.groupby(
                "Neighborhood",
                as_index=False
            )["SalePrice"]
            .mean()
            .sort_values(
                "SalePrice",
                ascending=False
            )
        )

        fig_bar = px.bar(
            promedios,
            x="Neighborhood",
            y="SalePrice",
            title="Precio promedio por vecindario",
            color="SalePrice"
        )

        fig_bar.update_layout(
            xaxis_title="Vecindario",
            yaxis_title="Precio promedio"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    c3, c4 = st.columns(2)

    with c3:
        fig_box = px.box(
            dff,
            x="Overall Qual",
            y="SalePrice",
            title="Precio según calidad general"
        )

        fig_box.update_layout(
            xaxis_title="Calidad general",
            yaxis_title="Precio de venta"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    with c4:
        fig_scatter = px.scatter(
            dff,
            x="Gr Liv Area",
            y="SalePrice",
            title="Área habitable vs precio de venta"
        )

        datos_tendencia = dff[
            ["Gr Liv Area", "SalePrice"]
        ].dropna()

        if (
            len(datos_tendencia) >= 2
            and datos_tendencia["Gr Liv Area"].nunique() >= 2
        ):
            valores_x = datos_tendencia[
                "Gr Liv Area"
            ].to_numpy()

            valores_y = datos_tendencia[
                "SalePrice"
            ].to_numpy()

            pendiente, intercepto = np.polyfit(
                valores_x,
                valores_y,
                1
            )

            linea_x = np.linspace(
                valores_x.min(),
                valores_x.max(),
                100
            )

            linea_y = (
                pendiente * linea_x
                + intercepto
            )

            fig_scatter.add_scatter(
                x=linea_x,
                y=linea_y,
                mode="lines",
                name="Línea de tendencia"
            )

        fig_scatter.update_layout(
            xaxis_title="Área habitable",
            yaxis_title="Precio de venta"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )


with tab2:
    st.header("Mapa interactivo de Panamá")

    st.write(
        "Esta sección utiliza datos sociodemográficos de Panamá. "
        "La variable representada es el total de viviendas por "
        "provincia, integrada con límites geográficos en formato "
        "GeoJSON."
    )

    fig_mapa = px.choropleth_mapbox(
        df_panama,
        geojson=geojson_panama,
        locations="provincia_limpia",
        featureidkey="properties.provincia_limpia",
        color="viviendas_total",
        hover_name="provincia",
        hover_data={
            "provincia_limpia": False,
            "viviendas_total": ":,.0f"
        },
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        center={
            "lat": 8.5,
            "lon": -80.0
        },
        zoom=6.2,
        opacity=0.75,
        title="Viviendas totales por provincia en Panamá"
    )

    fig_mapa.update_layout(
        margin={
            "r": 0,
            "t": 50,
            "l": 0,
            "b": 0
        },
        coloraxis_colorbar_title="Viviendas"
    )

    st.plotly_chart(
        fig_mapa,
        use_container_width=True
    )

    st.subheader("Gráfica resumen")

    fig_barras_panama = px.bar(
        df_panama.sort_values(
            "viviendas_total",
            ascending=False
        ),
        x="provincia",
        y="viviendas_total",
        color="viviendas_total",
        color_continuous_scale="Blues",
        title="Total de viviendas por provincia"
    )

    fig_barras_panama.update_layout(
        xaxis_title="Provincia",
        yaxis_title="Viviendas totales",
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_barras_panama,
        use_container_width=True
    )


with tab3:
    st.header("Simulador de precio de venta")

    st.write(
        "Ingrese las características de una vivienda para estimar "
        "su precio de venta utilizando el modelo de regresión "
        "seleccionado."
    )

    col1, col2 = st.columns(2)

    with col1:
        qual = st.slider(
            "Calidad General",
            min_value=1,
            max_value=10,
            value=6
        )

        area = st.number_input(
            "Área Habitable",
            min_value=0,
            max_value=6000,
            value=1500,
            step=50
        )

        cars = st.slider(
            "Capacidad del Garaje",
            min_value=0,
            max_value=4,
            value=2
        )

        g_area = st.slider(
            "Área del Garaje",
            min_value=0,
            max_value=1500,
            value=450,
            step=50
        )

    with col2:
        bsmt = st.number_input(
            "Área del Sótano",
            min_value=0,
            max_value=7000,
            value=1000,
            step=50
        )

        year = st.slider(
            "Año de Construcción",
            min_value=1880,
            max_value=2010,
            value=1985,
            step=5
        )

        baths = st.slider(
            "Baños Completos",
            min_value=0,
            max_value=4,
            value=2
        )

        beds = st.slider(
            "Dormitorios",
            min_value=0,
            max_value=6,
            value=3
        )

    if st.button(
        "Estimar precio de venta",
        type="primary"
    ):
        if modelo_reg is None:
            st.error(
                "No se pudo cargar el modelo de regresión."
            )

        else:
            try:
                vector = pd.DataFrame(
                    [
                        [
                            qual,
                            area,
                            cars,
                            g_area,
                            bsmt,
                            year,
                            baths,
                            beds
                        ]
                    ],
                    columns=variables_modelo
                )

                prediccion = modelo_reg.predict(
                    vector
                )[0]

                prediccion = float(prediccion)

                if np.isfinite(prediccion):
                    st.success(
                        "Precio estimado de venta: "
                        f"${prediccion:,.2f}"
                    )

                else:
                    st.error(
                        "El modelo no pudo generar una "
                        "predicción válida."
                    )

            except Exception:
                st.error(
                    "No fue posible generar la predicción. "
                    "Revise los valores ingresados e inténtelo "
                    "nuevamente."
                )
