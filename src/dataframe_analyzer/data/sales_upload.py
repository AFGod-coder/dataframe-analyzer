"""
Validation and normalization of a sales file uploaded by an end user.

The forecasting models in this project expect, for each store-product
pair, a CONTINUOUS DAILY series (one row per calendar day): Random
Forest/XGBoost build lag features with `shift()`, and the LSTM builds
sliding windows, both of which assume consecutive rows are consecutive
days. Real sales exports rarely look like that -- they usually have one
row per sale (several rows per day), skip days with no sales, use their
own column names and mix date formats. This module turns such a file
into the exact schema the rest of the project already uses
(`date, store, item, sales`), and reports every problem it finds in
plain language so the user can fix their file.

Expected file structure (documented to the user in the app, and shipped
as a downloadable template by `build_template_excel()`):

    | fecha      | tienda    | producto     | cantidad_vendida |
    |------------|-----------|--------------|------------------|
    | 2025-01-02 | Centro    | Arroz 1kg    | 14               |

    - fecha (required): sale date. Accepts YYYY-MM-DD, DD/MM/YYYY or a
      native Excel date cell.
    - producto (required): product name or code.
    - cantidad_vendida (required): units sold (number >= 0).
    - tienda (optional): store/branch name. If missing, every row is
      assigned to a single store.

Column names are matched case-insensitively, ignoring accents and
spaces, and common synonyms are accepted (e.g. "date", "sku", "ventas",
"unidades", "sucursal").
"""

import io
import logging
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum days of history (first to last date, after filling gaps) a
# product needs before it can be forecast. Below this, the Ensemble
# cannot carve out a separate validation window for its fusion weights
# AND still leave the lag/window-based models (lags up to 30 days, LSTM
# window of 30 days) enough rows to train on.
MIN_HISTORY_DAYS = 120

# Recommended history. Prophet fits yearly seasonality only when a
# training slice spans >= 365 days, and the Ensemble trains its base
# models on slices shorter than the full history (minus the evaluation
# window and the fusion-weight validation window, up to 90 days each).
# With 2 years, every one of those slices still covers a full year.
RECOMMENDED_HISTORY_DAYS = 730

DEFAULT_STORE_NAME = "Única"

# canonical column -> accepted synonyms (already normalized: lowercase,
# no accents, spaces/hyphens as underscores)
COLUMN_ALIASES = {
    "date": ["fecha", "date", "dia", "fecha_venta", "fecha_de_venta"],
    "item": ["producto", "item", "product", "sku", "articulo", "referencia", "codigo_producto"],
    "sales": [
        "cantidad_vendida", "cantidad", "ventas", "sales", "unidades",
        "unidades_vendidas", "cantidad_vendidas", "qty", "quantity",
    ],
    "store": ["tienda", "store", "sucursal", "local", "punto_de_venta", "sede"],
}
REQUIRED_COLUMNS = ["date", "item", "sales"]

SPANISH_NAMES = {
    "date": "fecha",
    "item": "producto",
    "sales": "cantidad_vendida",
    "store": "tienda",
}


class SalesFileError(ValueError):
    """Raised when the uploaded file cannot be turned into a usable dataset.

    The message is meant to be shown directly to a non-technical user.
    """


@dataclass
class UploadReport:
    """Everything the app needs to show after processing an uploaded file.

    Attributes:
        data (pd.DataFrame): Normalized daily dataset with columns
            date, store, item, sales (one row per store-item-day, gaps
            filled with 0).
        products (pd.DataFrame): One row per store-item pair with its
            history summary and whether it can be forecast.
        warnings (list[str]): Non-blocking issues, in plain Spanish.
        rows_read (int): Rows in the original file.
        rows_discarded (int): Rows dropped because of invalid values.
        duplicate_rows_aggregated (int): Rows merged because they shared
            the same day, store and product (e.g. one row per sale).
        days_filled (int): Calendar days added with 0 sales.
    """
    data: pd.DataFrame
    products: pd.DataFrame
    warnings: list = field(default_factory=list)
    rows_read: int = 0
    rows_discarded: int = 0
    duplicate_rows_aggregated: int = 0
    days_filled: int = 0


def _normalize_name(name) -> str:
    """Lowercase, strip accents, and turn spaces/hyphens into underscores."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return "_".join(text.strip().lower().replace("-", " ").split())


def read_sales_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded .xlsx or .csv file into a raw DataFrame.

    Args:
        file_bytes (bytes): Raw content of the uploaded file.
        filename (str): Original file name, used to pick the reader.

    Returns:
        pd.DataFrame: The file content, columns untouched.

    Raises:
        SalesFileError: If the format is unsupported or the file cannot be read.
    """
    name = filename.lower()
    try:
        if name.endswith(".xlsx"):
            # The template's instructions live in a second sheet, so
            # always read the first one.
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
        if name.endswith(".csv"):
            text = file_bytes.decode("utf-8-sig", errors="replace")
            # Excel in Spanish locales often exports CSV with ';'.
            sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
            return pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:  # pragma: no cover - depends on the user's file
        raise SalesFileError(f"No se pudo leer el archivo: {exc}") from exc
    raise SalesFileError("Formato no soportado. Sube un archivo Excel (.xlsx) o CSV (.csv).")


def _map_columns(raw: pd.DataFrame) -> dict:
    """Return {original_column: canonical_column} for recognized columns.

    Raises:
        SalesFileError: If a required column is missing or ambiguous.
    """
    normalized = {col: _normalize_name(col) for col in raw.columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        matches = [col for col, norm in normalized.items() if norm in aliases]
        if len(matches) > 1:
            raise SalesFileError(
                f"Hay más de una columna que parece ser '{SPANISH_NAMES[canonical]}': "
                f"{matches}. Deja solo una."
            )
        if matches:
            mapping[matches[0]] = canonical

    missing = [c for c in REQUIRED_COLUMNS if c not in mapping.values()]
    if missing:
        expected = ", ".join(SPANISH_NAMES[c] for c in missing)
        found = ", ".join(str(c) for c in raw.columns)
        raise SalesFileError(
            f"Faltan columnas obligatorias: {expected}. "
            f"Columnas encontradas en tu archivo: {found}. "
            "Descarga la plantilla para ver la estructura esperada."
        )
    return mapping


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates written as Excel dates, ISO (YYYY-MM-DD) or DD/MM/YYYY."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.astype(str).str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    iso_time = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    latin = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    latin_dash = pd.to_datetime(text, format="%d-%m-%Y", errors="coerce")
    return iso.fillna(iso_time).fillna(latin).fillna(latin_dash)


def _parse_quantities(series: pd.Series) -> pd.Series:
    """Parse quantities, accepting decimal commas written as text.

    "14", "14.5" and 14 are read as-is. A value containing a comma is
    read in the Spanish style: "1.234,5" -> 1234.5 and "14,5" -> 14.5.
    """
    # Checked as "is numeric" rather than "is object": in pandas >= 3 text
    # columns use the dedicated `str` dtype instead of `object`.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    has_comma = text.str.contains(",", regex=False)
    text = text.where(~has_comma, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def normalize_sales(raw: pd.DataFrame) -> UploadReport:
    """Validate a raw sales table and turn it into a continuous daily dataset.

    Steps:
        1. Map the user's column names to date/store/item/sales.
        2. Parse dates and quantities; discard rows that cannot be parsed
           (reported as a warning, not a crash).
        3. Aggregate rows sharing the same day, store and product (so a
           transaction-level export works as-is).
        4. Fill missing calendar days with 0 sales, per store-product,
           between its first and last recorded date.
        5. Summarize each store-product pair and flag whether it has
           enough history to be forecast.

    Args:
        raw (pd.DataFrame): The table as read from the user's file.

    Returns:
        UploadReport: normalized data + per-product summary + warnings.

    Raises:
        SalesFileError: If nothing usable remains after validation.
    """
    if raw is None or raw.empty:
        raise SalesFileError("El archivo está vacío.")

    warnings = []
    mapping = _map_columns(raw)
    df = raw[list(mapping)].rename(columns=mapping).copy()
    rows_read = len(df)

    if "store" not in df.columns:
        df["store"] = DEFAULT_STORE_NAME
    df["store"] = df["store"].fillna(DEFAULT_STORE_NAME).astype(str).str.strip()
    df["item"] = df["item"].astype(str).str.strip()
    df.loc[df["item"].isin(["", "nan", "None"]), "item"] = pd.NA

    df["date"] = _parse_dates(df["date"])
    df["sales"] = _parse_quantities(df["sales"])

    bad_date = df["date"].isna()
    bad_item = df["item"].isna()
    bad_sales = df["sales"].isna()
    negative = df["sales"] < 0
    invalid = bad_date | bad_item | bad_sales | negative

    if bad_date.any():
        warnings.append(f"{int(bad_date.sum())} fila(s) con fecha no reconocida fueron descartadas.")
    if bad_item.any():
        warnings.append(f"{int(bad_item.sum())} fila(s) sin producto fueron descartadas.")
    if bad_sales.any():
        warnings.append(f"{int(bad_sales.sum())} fila(s) con cantidad no numérica fueron descartadas.")
    if negative.any():
        warnings.append(
            f"{int(negative.sum())} fila(s) con cantidad negativa (posibles devoluciones) fueron descartadas."
        )

    df = df[~invalid]
    if df.empty:
        raise SalesFileError(
            "Ninguna fila tiene datos válidos. Revisa que las fechas, productos "
            "y cantidades estén en el formato de la plantilla."
        )

    df["date"] = df["date"].dt.normalize()
    before_agg = len(df)
    df = df.groupby(["store", "item", "date"], as_index=False)["sales"].sum()
    aggregated = before_agg - len(df)
    if aggregated:
        warnings.append(
            f"Se sumaron {aggregated} fila(s) que compartían fecha, tienda y producto "
            "(por ejemplo, una fila por cada venta del día)."
        )

    filled_parts = []
    days_filled = 0
    for (store, item), group in df.groupby(["store", "item"]):
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        series = group.set_index("date")["sales"].reindex(full_range, fill_value=0.0)
        days_filled += int(len(full_range) - len(group))
        part = series.rename_axis("date").reset_index()
        part["store"] = store
        part["item"] = item
        filled_parts.append(part)
    data = pd.concat(filled_parts, ignore_index=True)[["date", "store", "item", "sales"]]
    if days_filled:
        warnings.append(
            f"Se completaron {days_filled} día(s) sin registros con 0 ventas, "
            "porque los modelos necesitan una serie diaria continua."
        )

    products = summarize_products(data)
    if not products["apto"].any():
        raise SalesFileError(
            f"Ningún producto tiene al menos {MIN_HISTORY_DAYS} días de historia. "
            "Los modelos necesitan más datos para aprender patrones semanales y de tendencia."
        )
    short = products[products["apto"] & (products["dias_historia"] < RECOMMENDED_HISTORY_DAYS)]
    if not short.empty:
        warnings.append(
            f"{len(short)} producto(s) tienen menos de 2 años de historia: el modelo "
            "capturará la tendencia y el patrón semanal, pero puede no aprender bien la "
            "estacionalidad anual (temporadas altas y bajas del año), así que el "
            "pronóstico será menos preciso."
        )

    return UploadReport(
        data=data,
        products=products,
        warnings=warnings,
        rows_read=rows_read,
        rows_discarded=int(invalid.sum()),
        duplicate_rows_aggregated=int(aggregated),
        days_filled=days_filled,
    )


def summarize_products(data: pd.DataFrame) -> pd.DataFrame:
    """One row per store-product pair with its history summary.

    Args:
        data (pd.DataFrame): Normalized daily dataset (date, store, item, sales).

    Returns:
        pd.DataFrame: columns tienda, producto, fecha_inicio, fecha_fin,
        dias_historia, unidades_totales, promedio_diario,
        dias_con_venta_pct, apto.
    """
    grouped = data.groupby(["store", "item"])
    summary = grouped.agg(
        fecha_inicio=("date", "min"),
        fecha_fin=("date", "max"),
        dias_historia=("date", "count"),
        unidades_totales=("sales", "sum"),
        promedio_diario=("sales", "mean"),
        dias_con_venta=("sales", lambda s: (s > 0).mean() * 100),
    ).reset_index()
    summary = summary.rename(columns={"store": "tienda", "item": "producto", "dias_con_venta": "dias_con_venta_pct"})
    summary["promedio_diario"] = summary["promedio_diario"].round(2)
    summary["dias_con_venta_pct"] = summary["dias_con_venta_pct"].round(1)
    summary["apto"] = summary["dias_historia"] >= MIN_HISTORY_DAYS
    return summary.sort_values(["apto", "unidades_totales"], ascending=[False, False]).reset_index(drop=True)


def process_upload(file_bytes: bytes, filename: str) -> UploadReport:
    """Convenience wrapper: read + normalize in one call."""
    return normalize_sales(read_sales_file(file_bytes, filename))


def build_template_excel() -> bytes:
    """Build the downloadable Excel template (example data + instructions).

    Returns:
        bytes: An .xlsx file with a 'ventas' sheet (example rows in the
        exact expected structure) and an 'instrucciones' sheet.
    """
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    example = pd.DataFrame({
        "fecha": list(dates.strftime("%Y-%m-%d")) * 2,
        "tienda": ["Centro"] * 12,
        "producto": ["Arroz 1kg"] * 6 + ["Aceite 1L"] * 6,
        "cantidad_vendida": [14, 11, 16, 20, 25, 30, 6, 4, 7, 8, 9, 12],
    })
    instructions = pd.DataFrame({
        "columna": ["fecha", "tienda", "producto", "cantidad_vendida"],
        "obligatoria": ["Sí", "No", "Sí", "Sí"],
        "descripción": [
            "Fecha de la venta. Formatos: AAAA-MM-DD, DD/MM/AAAA o celda de fecha de Excel.",
            "Nombre de la tienda o sucursal. Si la omites, todo se trata como una sola tienda.",
            "Nombre o código del producto (debe escribirse igual en todas las filas).",
            "Unidades vendidas (número mayor o igual a 0).",
        ],
    })
    notes = pd.DataFrame({
        "recomendaciones": [
            f"Incluye al menos {MIN_HISTORY_DAYS} días de historia por producto "
            f"(ideal: {RECOMMENDED_HISTORY_DAYS} días o más para capturar la estacionalidad anual).",
            "Puedes subir una fila por día o una fila por cada venta: las filas del mismo día se suman.",
            "Los días sin ninguna fila se asumen como 0 ventas.",
            "Borra las filas de ejemplo antes de pegar tus datos. Deja los encabezados tal cual.",
        ]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        example.to_excel(writer, sheet_name="ventas", index=False)
        instructions.to_excel(writer, sheet_name="instrucciones", index=False)
        notes.to_excel(writer, sheet_name="instrucciones", index=False, startrow=len(instructions) + 2)
        for sheet in writer.sheets.values():
            for column_cells in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(12, width + 2), 90)
    return buffer.getvalue()


def project_dataset_to_template(df_project: pd.DataFrame, store: int, items: list, last_days: int = 730) -> pd.DataFrame:
    """Turn a slice of the project's Kaggle dataset into the upload template format.

    Used by the app's "use example data" button, so the upload flow can
    be demonstrated without preparing a file by hand.

    Args:
        df_project (pd.DataFrame): Project dataset (date, store, item, sales).
        store (int): Store to take.
        items (list): Item ids to take.
        last_days (int): How many trailing days of history to keep.

    Returns:
        pd.DataFrame: columns fecha, tienda, producto, cantidad_vendida.
    """
    subset = df_project[(df_project["store"] == store) & (df_project["item"].isin(items))].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset[subset["date"] > subset["date"].max() - pd.Timedelta(days=last_days)]
    return pd.DataFrame({
        "fecha": subset["date"].dt.strftime("%Y-%m-%d"),
        "tienda": "Tienda " + subset["store"].astype(str),
        "producto": "Producto " + subset["item"].astype(str),
        "cantidad_vendida": subset["sales"],
    }).reset_index(drop=True)
