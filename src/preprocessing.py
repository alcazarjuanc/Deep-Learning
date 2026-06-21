from pathlib import Path
import ast

import numpy as np
import pandas as pd


def parse_list_column(value):
    """
    Convierte columnas guardadas como texto en listas de Python.
    Ejemplo: "['Museo', 'Cultura']" -> ['Museo', 'Cultura']
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def unique_sorted_list(list_of_lists):
    """
    Une listas, elimina duplicados y devuelve una lista ordenada.
    """
    items = []

    for sublist in list_of_lists:
        if isinstance(sublist, list):
            items.extend(sublist)

    return sorted(list(set(items)))


def build_image_path(row, project_root):
    """
    Construye la ruta local de la imagen usando main_image_path.
    """
    image_path = Path(row["main_image_path"])

    if image_path.is_absolute():
        return image_path

    return Path(project_root) / image_path


def add_image_paths(df, project_root):
    """
    Agrega columnas para la ruta local de la imagen y verificación de existencia.
    """
    df = df.copy()

    df["image_path_local"] = df.apply(
        lambda row: build_image_path(row, project_root),
        axis=1
    )

    df["image_exists"] = df["image_path_local"].apply(lambda path: Path(path).exists())

    return df


def prepare_list_columns(df):
    """
    Convierte categories y tags a listas reales de Python.
    """
    df = df.copy()

    df["categories_list"] = df["categories"].apply(parse_list_column)
    df["tags_list"] = df["tags"].apply(parse_list_column)

    return df


def aggregate_pois(df):
    """
    Agrupa el dataset por id para trabajar con POIs únicos.
    """
    agg_dict = {
        "name": "first",
        "shortDescription": "first",
        "tier": "first",
        "locationLon": "first",
        "locationLat": "first",
        "xps": "mean",
        "Visits": "mean",
        "Likes": "mean",
        "Dislikes": "mean",
        "Bookmarks": "mean",
        "main_image_path": "first",
        "image_path_local": "first",
        "image_exists": "first",
        "categories_list": unique_sorted_list,
        "tags_list": unique_sorted_list,
    }

    df_model = (
        df
        .groupby("id", as_index=False)
        .agg(agg_dict)
        .reset_index(drop=True)
    )

    return df_model


def add_engagement_score(df):
    """
    Construye la métrica de engagement usada en la práctica.
    """
    df = df.copy()

    df["engagement_score"] = (
        0.25 * np.log1p(df["Visits"]) +
        2.00 * np.log1p(df["Likes"]) +
        2.00 * np.log1p(df["Bookmarks"]) -
        1.00 * np.log1p(df["Dislikes"])
    )

    return df


def add_binary_engagement_label(df):
    """
    Crea la etiqueta binaria de engagement usando la mediana como umbral.
    """
    df = df.copy()

    threshold = df["engagement_score"].median()

    df["engagement_label"] = (
        df["engagement_score"] >= threshold
    ).astype(int)

    return df, threshold