from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Seoul Dong Industry Recommender")

# ---------------------------
# CORS (프론트가 다른 포트에서 호출해도 허용)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Paths (프로젝트 폴더에 있어야 함)
# ---------------------------
MODEL_GROWTH_PATH = Path("model_growth.pkl")
MODEL_RISK_PATH = Path("model_risk.pkl")
FEATURES_PATH = Path("features_panel.parquet")
BIZMAP_PATH = Path("biz_code_map.csv")
DONGMAP_PATH = Path("dong_map.csv")
# ---------------------------
# Load artifacts
# ---------------------------
model_growth = joblib.load(MODEL_GROWTH_PATH)
model_risk = joblib.load(MODEL_RISK_PATH)
features_panel = pd.read_parquet(FEATURES_PATH)

biz_map = None
if BIZMAP_PATH.exists():
    biz_map = pd.read_csv(BIZMAP_PATH)
    biz_map["biz_code"] = biz_map["biz_code"].astype(str).str.strip()
    if "biz_name" in biz_map.columns:
        biz_map["biz_name"] = biz_map["biz_name"].astype(str).str.strip()

dong_map = None
if DONGMAP_PATH.exists():
    dong_map = pd.read_csv(DONGMAP_PATH)
    dong_map["dong_code"] = dong_map["dong_code"].astype(str).str.strip()
    dong_map["dong_name"] = dong_map["dong_name"].astype(str).str.strip()

# ---------------------------
# Utils
# ---------------------------
def make_reasons(row: pd.Series, quarter_df: pd.DataFrame) -> list[str]:
    """
    아주 단순한 '추천 이유' 생성.
    - 같은 분기 전체의 중앙값과 비교해서 수요/경쟁/안정성 설명.
    """
    reasons = []

    def med(col):
        return quarter_df[col].median() if col in quarter_df.columns else np.nan

    # 수요
    if "pop_mean" in row.index and "pop_mean" in quarter_df.columns and pd.notna(row["pop_mean"]):
        if row["pop_mean"] >= med("pop_mean"):
            reasons.append("생활인구 평균이 높은 편(수요 유리)")
        else:
            reasons.append("생활인구 평균이 낮은 편(수요 약함)")

    if "store_cnt" in row.index and "store_cnt" in quarter_df.columns and pd.notna(row["store_cnt"]):
        if row["store_cnt"] <= med("store_cnt"):
            reasons.append("동일 업종 점포 수가 상대적으로 적음(경쟁 완화)")
        else:
            reasons.append("동일 업종 점포 수가 많은 편(경쟁 치열)")

    if "close_rate" in row.index and "close_rate" in quarter_df.columns and pd.notna(row["close_rate"]):
        if row["close_rate"] <= med("close_rate"):
            reasons.append("현재 폐업률이 낮은 편(안정적)")
        else:
            reasons.append("현재 폐업률이 높은 편(리스크 주의)")

    return reasons[:3]


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/quarters")
def list_quarters():
    qs = sorted(features_panel["quarter"].astype(str).unique().tolist())
    return {"quarters": qs}

@app.get("/dong_codes")
def list_dong_codes(quarter: str = Query(...)):
    dfq = features_panel[features_panel["quarter"].astype(str) == str(quarter)]
    codes = sorted(dfq["dong_code"].astype(str).unique().tolist())
    return {"dong_codes": codes}

@app.get("/search_dong")
def search_dong(
    q: str = Query(..., description="행정동 코드(부분) 또는 동 이름(있으면)"),
    quarter: str = Query(None, description="분기 필터 (예: 2024Q3)")
):
    """
    - dong_map.csv가 있으면 '동 이름 검색' 지원
    - 없으면 '코드 검색(부분 포함)'만 지원
    """
    q = str(q).strip()
    if not q:
        return {"items": []}

    df = features_panel
    if quarter:
        df = df[df["quarter"].astype(str) == str(quarter)]

    codes = df["dong_code"].astype(str).unique()

    items = []

    for code in codes:
        if q in code:
            items.append({"dong_code": code, "dong_name": None})

    if dong_map is not None:
        valid = set(codes.tolist())
        sub = dong_map[dong_map["dong_code"].isin(valid)].copy()

        # 이름 포함 검색
        hit = sub[sub["dong_name"].str.contains(q, case=False, na=False)]
        for _, r in hit.iterrows():
            items.append({"dong_code": r["dong_code"], "dong_name": r["dong_name"]})

        exact = sub[sub["dong_code"] == q]
        for _, r in exact.iterrows():
            items.append({"dong_code": r["dong_code"], "dong_name": r["dong_name"]})

    seen = set()
    unique = []
    for it in items:
        code = it["dong_code"]
        if code in seen:
            continue
        seen.add(code)
        unique.append(it)
        if len(unique) >= 30:
            break

    return {"items": unique}

@app.get("/recommend")
def recommend(
    dong_code: str = Query(..., description="행정동코드"),
    quarter: str = Query(..., description="입력 분기 (예: 2024Q3)"),
    top_n: int = Query(10, ge=1, le=50),
    alpha: float = Query(1.0, ge=0.0, le=5.0),
):
    """
    점수 = proba_growth - alpha * proba_risk
    """
    subset = features_panel[
        (features_panel["dong_code"].astype(str) == str(dong_code)) &
        (features_panel["quarter"].astype(str) == str(quarter))
    ].copy()

    if subset.empty:
        return {"dong_code": dong_code, "quarter": quarter, "items": [], "message": "해당 행정동/분기 데이터가 없습니다."}

    quarter_df = features_panel[features_panel["quarter"].astype(str) == str(quarter)]

    proba_g = model_growth.predict_proba(subset)[:, 1]
    proba_r = model_risk.predict_proba(subset)[:, 1]

    subset["proba_growth"] = proba_g
    subset["proba_risk"] = proba_r
    subset["score"] = subset["proba_growth"] - alpha * subset["proba_risk"]

    if biz_map is not None:
        subset = subset.merge(biz_map, on="biz_code", how="left")
    else:
        subset["biz_name"] = None

    subset = subset.sort_values("score", ascending=False).head(top_n)

    items = []
    for _, row in subset.iterrows():
        reasons = make_reasons(row, quarter_df)
        items.append({
            "biz_code": str(row["biz_code"]),
            "biz_name": (None if pd.isna(row.get("biz_name")) else row.get("biz_name")),
            "score": float(row["score"]),
            "proba_growth": float(row["proba_growth"]),
            "proba_risk": float(row["proba_risk"]),
            "reasons": reasons
        })

    dong_name = None
    if dong_map is not None:
        m = dong_map[dong_map["dong_code"].astype(str) == str(dong_code)]
        if not m.empty:
            dong_name = m.iloc[0]["dong_name"]

    return {"dong_code": dong_code, "dong_name": dong_name, "quarter": quarter, "alpha": alpha, "items": items}
