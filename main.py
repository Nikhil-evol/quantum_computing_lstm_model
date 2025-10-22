
from fastapi import FastAPI, Request
import datetime
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import pandas as pd

app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

COMPANIES = ["SBUX"]

# Model utilities are imported lazily inside the prediction endpoint
model = None
weights = None
scaler = None

class StockData(BaseModel):
    recent_closes: list[float]

@app.get("/")
def form(request: Request, company: str = "SBUX"):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year}
    )

@app.post("/predict_form")
async def predict_form(request: Request):
    try:
        form = await request.form()
        company = form.get('company', 'SBUX')
        days = int(form.get('days', 7))

        # Lazy import to avoid startup failures
        from model_utils import load_model_and_scaler, predict_n_days
        global model, weights, scaler

        if model is None or weights is None or scaler is None:
            try:
                model, scaler, weights = load_model_and_scaler()
            except Exception as e:
                return templates.TemplateResponse(
                    "index.html",
                    {"request": request, "error": f"Model load failed: {e}", 
                     "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year}
                )

        # Load CSV
        csv_path = os.path.join(os.path.dirname(__file__), f"{company}.csv")
        if not os.path.exists(csv_path):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"CSV file not found at {csv_path}", 
                 "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year}
            )

        df = pd.read_csv(csv_path, index_col=0)
        # Ensure proper datetime parsing
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
        df.ffill(inplace=True)  # fill missing dates

        if 'Close' not in df.columns:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "CSV does not contain 'Close' column.",
                 "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year}
            )
        df = df[['Close']]

        print("✅ CSV loaded successfully. Date range:", df.index.min(), "to", df.index.max())

        closes = df["Close"].dropna().tolist()
        window_size = 4
        if len(closes) < window_size:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Insufficient data: need at least {window_size} closes.",
                 "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year}
            )

        recent = closes[-window_size:]
        preds = predict_n_days(model=model, scaler=scaler, q_weights=weights,
                               recent_data=recent, window_size=window_size, n_days=days)
        preds_rounded = [f"{x:.2f}" for x in preds]

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "predictions": preds_rounded, "companies": COMPANIES,
             "company": company, "year": datetime.datetime.now().year}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "companies": COMPANIES,
             "company": company, "year": datetime.datetime.now().year}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)

