"""
UsedCarAI - app/prediction_app.py
Gradio interface for used-car listing-price estimation.
"""

from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import gradio as gr

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import (
    load_project,
    prepare_input,
    predict_all,
    format_thb,
    unique_options,
    evaluate_prediction_reliability,
)

PROJECT = load_project()
RF_MODEL = PROJECT["rf_model"]
LR_MODEL = PROJECT["lr_model"]
MARKET = PROJECT["market"]
CLEANED = PROJECT["cleaned"]

# Dataset-driven choices: do not invent categories that the models never saw.
def choices(col):
    if col not in CLEANED.columns:
        return []
    return sorted(
        CLEANED[col].dropna().astype(str).str.strip()
        .loc[lambda s: s.ne("")]
        .drop_duplicates().tolist()
    )

BRANDS = choices("brand")
FUELS = choices("fuel_type")
TRANSMISSIONS = choices("transmission")
BODY_TYPES = choices("body_type")
COLORS = choices("color")
PROVINCES = choices("province")
LOCATIONS = choices("location")
SELLER_TYPES = choices("seller_type")

years = pd.to_numeric(CLEANED.get("model_year"), errors="coerce").dropna()
YEAR_MIN = int(years.min()) if len(years) else 1980
YEAR_MAX = int(years.max()) if len(years) else 2026

engines = pd.to_numeric(CLEANED.get("engine_size"), errors="coerce").dropna()
ENGINE_MEDIAN = float(engines.median()) if len(engines) else 1500.0

seats = pd.to_numeric(CLEANED.get("number_of_seats"), errors="coerce").dropna()
SEAT_MEDIAN = int(round(seats.median())) if len(seats) else 5


def update_models(brand):
    opts = unique_options(CLEANED, "model", brand=brand) if brand else []
    return gr.update(choices=opts, value=opts[0] if opts else None), gr.update(choices=[], value=None)


def update_submodels(brand, model):
    opts = unique_options(CLEANED, "sub_model", brand=brand, model=model) if brand and model else []
    return gr.update(choices=opts, value=opts[0] if opts else None)


def update_locations(province):
    if not province:
        return gr.update(choices=LOCATIONS, value=None)
    opts = unique_options(CLEANED, "location", province=province)
    return gr.update(choices=opts, value=opts[0] if opts else None)


def estimate(
    brand, model, sub_model, model_year, mileage,
    fuel_type, transmission, engine_size, body_type,
    color, province, location, seller_type, number_of_seats
):
    try:
        if not brand or not model:
            raise ValueError("กรุณาเลือก Brand และ Model")
        if model_year is None:
            raise ValueError("กรุณาระบุปีรถ")
        if mileage is None:
            raise ValueError("กรุณาระบุเลขไมล์")

        x = prepare_input(
            brand=brand,
            model=model,
            sub_model=sub_model,
            model_year=model_year,
            mileage=mileage,
            fuel_type=fuel_type,
            transmission=transmission,
            engine_size=engine_size,
            body_type=body_type,
            color=color,
            province=province,
            location=location,
            seller_type=seller_type,
            number_of_seats=number_of_seats,
            reference_year=2026,
        )

        result = predict_all(x, RF_MODEL, LR_MODEL, MARKET)

        reliability = evaluate_prediction_reliability(
            rf_price=result["random_forest_price"],
            lr_price=result["linear_regression_price"],
            hybrid_price=result["hybrid_price"],
            comparable_count=result["comparable_count"],
            similarity=result["comparable_similarity"],
        )

        warning_text = "\n".join(
            f"- {w}" for w in reliability["warnings"]
        )

        reliability_text = (
            f"ระดับความน่าเชื่อถือ: {reliability['level']}\n"
            f"Reliability Score: {reliability['score']}/100\n"
            f"Model disagreement: {reliability['model_disagreement']:.3f}\n\n"
            f"คำเตือน:\n{warning_text}"
        )

        comp_text = format_thb(result["comparable_price"])
        if result["hybrid_used"]:
            hybrid_note = (
                f"Hybrid ใช้ 50% Random Forest + 50% Comparable "
                f"({result['comparable_count']} คัน)"
            )
        else:
            hybrid_note = (
                f"Comparable ไม่เพียงพอ ({result['comparable_count']} คัน) "
                "จึง fallback เป็น Random Forest"
            )

        detail = (
            f"Comparable: {comp_text}\n"
            f"จำนวน Comparable: {result['comparable_count']}\n"
            f"Similarity เฉลี่ย: {result['comparable_similarity']:.3f}\n"
            f"Match level: {result['comparable_match_level']}\n"
            f"{hybrid_note}"
        )

        return (
            format_thb(result["random_forest_price"]),
            format_thb(result["linear_regression_price"]),
            format_thb(result["hybrid_price"]),
            detail,
            reliability_text,
            "คำนวณสำเร็จ"
        )

    except Exception as e:
        return "-", "-", "-", "", "", f"เกิดข้อผิดพลาด: {e}"


with gr.Blocks(title="UsedCarAI") as demo:
    gr.Markdown(
        """
# 🚗 UsedCarAI
### ระบบประมาณ **ราคาประกาศรถมือสอง (Listing Price)**

กรอกข้อมูลรถ แล้วระบบจะแสดงผลจาก Random Forest, Linear Regression และ Hybrid

> ผลลัพธ์เป็นค่าประมาณจากข้อมูลและโมเดลของโครงการ ไม่ใช่ราคาซื้อขายที่รับประกัน
"""
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ข้อมูลรถ")

            brand = gr.Dropdown(BRANDS, label="Brand", filterable=True)
            model = gr.Dropdown([], label="Model", filterable=True)
            sub_model = gr.Dropdown([], label="Sub-model", filterable=True)

            model_year = gr.Number(
                label=f"Model year ({YEAR_MIN}–{YEAR_MAX})",
                value=YEAR_MAX,
                precision=0
            )
            mileage = gr.Number(label="Mileage (km)", value=50000, minimum=0)

            fuel_type = gr.Dropdown(FUELS, label="Fuel type", filterable=True)
            transmission = gr.Dropdown(TRANSMISSIONS, label="Transmission", filterable=True)
            engine_size = gr.Number(label="Engine size", value=ENGINE_MEDIAN, minimum=0)
            body_type = gr.Dropdown(BODY_TYPES, label="Body type", filterable=True)

            color = gr.Dropdown(COLORS, label="Color", filterable=True)
            province = gr.Dropdown(PROVINCES, label="Province", filterable=True)
            location = gr.Dropdown(LOCATIONS, label="Location", filterable=True)
            seller_type = gr.Dropdown(SELLER_TYPES, label="Seller type", filterable=True)
            number_of_seats = gr.Number(
                label="Number of seats",
                value=SEAT_MEDIAN,
                minimum=1,
                precision=0
            )

            predict_btn = gr.Button("ประเมินราคา", variant="primary")
            clear_btn = gr.ClearButton()

        with gr.Column():
            gr.Markdown("### ผลการประมาณราคา")
            rf_output = gr.Textbox(label="Random Forest", interactive=False)
            lr_output = gr.Textbox(label="Linear Regression", interactive=False)
            hybrid_output = gr.Textbox(label="Hybrid", interactive=False)
            details = gr.Textbox(
                label="Hybrid / Comparable details",
                lines=6,
                interactive=False
            )

            reliability_output = gr.Textbox(
                label="Prediction Reliability",
                lines=7,
                interactive=False
            )

            status = gr.Textbox(label="Status", interactive=False)

    brand.change(
        update_models,
        inputs=brand,
        outputs=[model, sub_model]
    )
    model.change(
        update_submodels,
        inputs=[brand, model],
        outputs=sub_model
    )
    province.change(
        update_locations,
        inputs=province,
        outputs=location
    )

    inputs = [
        brand, model, sub_model, model_year, mileage,
        fuel_type, transmission, engine_size, body_type,
        color, province, location, seller_type, number_of_seats
    ]
    outputs = [
        rf_output,
        lr_output,
        hybrid_output,
        details,
        reliability_output,
        status
    ]

    predict_btn.click(estimate, inputs=inputs, outputs=outputs)
    clear_btn.add(inputs + outputs)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        debug=False,
    )
