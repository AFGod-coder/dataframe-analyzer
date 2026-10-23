"""Tests for the user sales-file validation/normalization (dataframe_analyzer.data.sales_upload)."""

import io

import numpy as np
import pandas as pd
import pytest

from dataframe_analyzer.data.sales_upload import (
    DEFAULT_STORE_NAME,
    MIN_HISTORY_DAYS,
    SalesFileError,
    build_template_excel,
    normalize_sales,
    process_upload,
)


def _daily_rows(product="Arroz", days=MIN_HISTORY_DAYS, start="2025-01-01", store=None, value=10):
    dates = pd.date_range(start, periods=days, freq="D")
    df = pd.DataFrame({"fecha": dates.strftime("%Y-%m-%d"), "producto": product, "cantidad_vendida": value})
    if store is not None:
        df["tienda"] = store
    return df


def test_accepts_synonyms_accents_and_missing_store():
    raw = _daily_rows().rename(columns={"fecha": "Fecha", "producto": "SKU", "cantidad_vendida": "Unidades Vendidas"})
    report = normalize_sales(raw)
    assert list(report.data.columns) == ["date", "store", "item", "sales"]
    assert (report.data["store"] == DEFAULT_STORE_NAME).all()
    assert report.products["apto"].all()


def test_missing_required_column_is_explained():
    raw = _daily_rows().drop(columns=["cantidad_vendida"])
    with pytest.raises(SalesFileError, match="cantidad_vendida"):
        normalize_sales(raw)


def test_transaction_rows_are_aggregated_per_day():
    raw = pd.concat([_daily_rows(value=3), _daily_rows(value=4)], ignore_index=True)
    report = normalize_sales(raw)
    assert report.duplicate_rows_aggregated == MIN_HISTORY_DAYS
    assert (report.data["sales"] == 7).all()


def test_missing_days_are_filled_with_zero():
    raw = _daily_rows()
    raw = raw.drop(index=[10, 11, 12]).reset_index(drop=True)
    report = normalize_sales(raw)
    assert report.days_filled == 3
    assert len(report.data) == MIN_HISTORY_DAYS
    full = report.data.set_index("date")["sales"]
    assert full[pd.Timestamp("2025-01-11")] == 0


def test_invalid_rows_are_discarded_with_warnings():
    raw = _daily_rows(days=MIN_HISTORY_DAYS + 3)
    raw["cantidad_vendida"] = raw["cantidad_vendida"].astype(object)  # mixed numbers/text, as in a real file
    raw.loc[0, "fecha"] = "no es fecha"
    raw.loc[1, "cantidad_vendida"] = "abc"
    raw.loc[2, "cantidad_vendida"] = -5
    report = normalize_sales(raw)
    assert report.rows_discarded == 3
    text = " ".join(report.warnings)
    assert "fecha no reconocida" in text and "no numérica" in text and "negativa" in text


def test_latin_dates_and_decimal_commas():
    raw = pd.DataFrame({
        "fecha": pd.date_range("2025-01-01", periods=MIN_HISTORY_DAYS).strftime("%d/%m/%Y"),
        "producto": "Aceite",
        "cantidad_vendida": ["1.234,5"] + ["14,5"] + ["14.5"] * (MIN_HISTORY_DAYS - 2),
    })
    report = normalize_sales(raw)
    sales = report.data.sort_values("date")["sales"].tolist()
    assert sales[:3] == [1234.5, 14.5, 14.5]
    assert report.data["date"].min() == pd.Timestamp("2025-01-01")


def test_short_history_products_are_not_eligible():
    raw = pd.concat([_daily_rows("Largo"), _daily_rows("Corto", days=30)], ignore_index=True)
    report = normalize_sales(raw)
    eligible = dict(zip(report.products["producto"], report.products["apto"]))
    assert eligible == {"Largo": True, "Corto": False}


def test_no_eligible_product_raises():
    with pytest.raises(SalesFileError, match="días de historia"):
        normalize_sales(_daily_rows(days=30))


def test_multiple_stores_are_independent_series():
    raw = pd.concat([_daily_rows(store="Centro"), _daily_rows(store="Norte", value=2)], ignore_index=True)
    report = normalize_sales(raw)
    assert set(report.data["store"]) == {"Centro", "Norte"}
    assert report.products.shape[0] == 2


def test_template_roundtrip_has_expected_structure():
    content = build_template_excel()
    sheet = pd.read_excel(io.BytesIO(content), sheet_name=0)
    assert list(sheet.columns) == ["fecha", "tienda", "producto", "cantidad_vendida"]
    instructions = pd.read_excel(io.BytesIO(content), sheet_name="instrucciones")
    assert "columna" in instructions.columns


def test_process_upload_reads_xlsx_and_semicolon_csv():
    raw = _daily_rows()
    xlsx = io.BytesIO()
    raw.to_excel(xlsx, index=False)
    assert process_upload(xlsx.getvalue(), "ventas.xlsx").products["apto"].all()

    csv = raw.to_csv(index=False, sep=";").encode("utf-8")
    assert process_upload(csv, "ventas.csv").products["apto"].all()

    with pytest.raises(SalesFileError, match="Formato no soportado"):
        process_upload(b"x", "ventas.pdf")
