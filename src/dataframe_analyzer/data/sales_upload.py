"""Check and clean a sales file that a user uploads.

The models need one daily series for each store and product, with one
row for every calendar day. Random Forest and XGBoost use lag features
made with `shift()`, and the LSTM uses sliding windows. Both expect that
two rows next to each other are two days next to each other.

Real sales files are usually different. They often have one row for each
sale, they skip days without sales, they use other column names and they
mix date formats. This module changes such a file into the same columns
that the rest of the project uses (`date`, `store`, `item`, `sales`). It
also reports every problem in simple words, so the user can fix the file.

Expected file structure (the app shows it, and `build_template_excel()`
creates a template that the user can download):

    | fecha      | tienda    | producto     | cantidad_vendida |
    |------------|-----------|--------------|------------------|
    | 2025-01-02 | Centro    | Arroz 1kg    | 14               |

    - `fecha` (required): date of the sale. Formats: YYYY-MM-DD,
      DD/MM/YYYY or an Excel date cell.
    - `producto` (required): product name or code.
    - `cantidad_vendida` (required): units sold (a number >= 0).
    - `tienda` (optional): store name. If the column is missing, all rows
      go to one store.

Column names are compared without case, accents or spaces, and common
other names are accepted (for example `date`, `sku`, `ventas`,
`unidades`, `sucursal`).
"""

import io
import logging
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of days of history (from the first to the last date,
# after filling gaps) that a product needs before we can forecast it.
# With less data, the ensemble cannot keep a separate validation window
# for its weights and still give the lag and window models (lags and LSTM
# window of up to 30 days) enough rows to train.
MIN_HISTORY_DAYS = 120

# Recommended history. Prophet uses yearly seasonality only when the
# training data covers at least 365 days. The ensemble trains its base
# models on less than the full history (it removes the test window and
# the weight validation window, up to 90 days each). With 2 years of
# data, every training part still covers a full year.
RECOMMENDED_HISTORY_DAYS = 730

DEFAULT_STORE_NAME = "Única"

# Standard column name -> accepted names (already normalized: lowercase,
# no accents, spaces and hyphens changed to underscores).
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
    """Error raised when the uploaded file cannot become a usable dataset.

    The message is written for a non-technical user, so the app can show
    it directly.
    """


@dataclass
class UploadReport:
    """Result of processing an uploaded file, ready to show in the app.

    Attributes:
        data (pd.DataFrame): Clean daily dataset with the columns `date`,
            `store`, `item` and `sales`. It has one row for each store,
            item and day, and missing days have 0 sales.
        products (pd.DataFrame): One row for each store and item, with a
            summary of its history and a flag that says if it can be
            forecast.
        warnings (list[str]): Problems that do not stop the process,
            written for the user in Spanish.
        rows_read (int): Number of rows in the original file.
        rows_discarded (int): Rows removed because of invalid values.
        duplicate_rows_aggregated (int): Rows added together because they
            had the same day, store and product (for example, one row for
            each sale).
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
    """Normalize a column name so it can be compared with known names.

    Args:
        name: Original column name. Any value is changed to text first.

    Returns:
        str: The name in lowercase, without accents, and with spaces and
        hyphens changed to underscores.
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return "_".join(text.strip().lower().replace("-", " ").split())


def read_sales_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded .xlsx or .csv file into a raw DataFrame.

    Args:
        file_bytes (bytes): Content of the uploaded file.
        filename (str): Original file name. Its extension decides how the
            file is read.

    Returns:
        pd.DataFrame: The file content, with the original columns.

    Raises:
        SalesFileError: If the format is not supported or the file cannot
            be read.
    """
    name = filename.lower()
    try:
        if name.endswith(".xlsx"):
            # The template keeps its instructions in the second sheet, so
            # we always read the first sheet.
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
        if name.endswith(".csv"):
            text = file_bytes.decode("utf-8-sig", errors="replace")
            # Excel with Spanish settings often saves CSV files with ';'.
            sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
            return pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:  # pragma: no cover - depends on the user's file
        raise SalesFileError(f"No se pudo leer el archivo: {exc}") from exc
    raise SalesFileError("Formato no soportado. Sube un archivo Excel (.xlsx) o CSV (.csv).")


def _map_columns(raw: pd.DataFrame) -> dict:
    """Find which columns of the file are the date, store, item and sales.

    Args:
        raw (pd.DataFrame): Table as read from the user's file.

    Returns:
        dict: Map from each recognized original column name to its
        standard name (`date`, `store`, `item` or `sales`).

    Raises:
        SalesFileError: If a required column is missing, or if more than
            one column matches the same standard name.
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
    """Convert a column of dates written in different formats.

    Accepted formats are Excel date cells, YYYY-MM-DD (with or without a
    time), DD/MM/YYYY and DD-MM-YYYY.

    Args:
        series (pd.Series): Column with the dates from the file.

    Returns:
        pd.Series: Dates as datetime values. Values that cannot be read
        are `NaT`.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.astype(str).str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    iso_time = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    latin = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    latin_dash = pd.to_datetime(text, format="%d-%m-%Y", errors="coerce")
    return iso.fillna(iso_time).fillna(latin).fillna(latin_dash)


def _parse_quantities(series: pd.Series) -> pd.Series:
    """Convert a column of quantities, also when they use decimal commas.

    The values "14", "14.5" and 14 are read as they are. A value with a
    comma is read in the Spanish style: "1.234,5" becomes 1234.5 and
    "14,5" becomes 14.5.

    Args:
        series (pd.Series): Column with the quantities from the file.

    Returns:
        pd.Series: Numbers as floats. Values that cannot be read are NaN.
    """
    # We check "is numeric" and not "is object" because in pandas 3 and
    # later, text columns use the `str` dtype instead of `object`.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    has_comma = text.str.contains(",", regex=False)
    text = text.where(~has_comma, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def normalize_sales(raw: pd.DataFrame) -> UploadReport:
    """Check a sales table and change it into a continuous daily dataset.

    Steps:
        1. Match the user's column names to `date`, `store`, `item` and
           `sales`.
        2. Read dates and quantities. Rows that cannot be read are
           removed and reported as a warning (the process does not stop).
        3. Add together rows with the same day, store and product, so a
           file with one row for each sale also works.
        4. For each store and product, add the missing calendar days
           between the first and the last date, with 0 sales.
        5. Summarize each store and product, and mark if it has enough
           history to be forecast.

    Args:
        raw (pd.DataFrame): Table as read from the user's file.

    Returns:
        UploadReport: Clean data, a summary for each product and the
        warnings.

    Raises:
        SalesFileError: If the table is empty, if no valid row remains, or
            if no product has enough history.
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
    """Build a history summary with one row for each store and product.

    Args:
        data (pd.DataFrame): Clean daily dataset with the columns `date`,
            `store`, `item` and `sales`.

    Returns:
        pd.DataFrame: Summary with the columns `tienda`, `producto`,
        `fecha_inicio`, `fecha_fin`, `dias_historia`, `unidades_totales`,
        `promedio_diario`, `dias_con_venta_pct` and `apto`. The column
        `apto` is True when the product has enough history. Rows that can
        be forecast come first, ordered by total units.
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
    """Read an uploaded file and clean it in one step.

    Args:
        file_bytes (bytes): Content of the uploaded file.
        filename (str): Original file name (.xlsx or .csv).

    Returns:
        UploadReport: Clean data, a summary for each product and the
        warnings.

    Raises:
        SalesFileError: If the file cannot be read or has no usable data.
    """
    return normalize_sales(read_sales_file(file_bytes, filename))


def build_template_excel() -> bytes:
    """Create the Excel template that the user can download.

    Returns:
        bytes: An .xlsx file with two sheets. The sheet `ventas` has
        example rows with the expected structure, and the sheet
        `instrucciones` explains each column.
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
    """Change part of the project dataset into the upload template format.

    The app uses it for the "use example data" button, so the user can
    try the upload flow without preparing a file.

    Args:
        df_project (pd.DataFrame): Project dataset with the columns
            `date`, `store`, `item` and `sales`.
        store (int): Store to take.
        items (list): Item ids to take.
        last_days (int): Number of most recent days of history to keep.

    Returns:
        pd.DataFrame: Table with the columns `fecha`, `tienda`, `producto`
        and `cantidad_vendida`.
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
