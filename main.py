from pathlib import Path
import py_compile

# 직전 셀에서 작성한 전체 코드 문자열을 그대로 다시 구성합니다.
code = r'''import json
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="전국 시군구별 출산율 지도",
    page_icon="👶",
    layout="wide",
)

POPULATION_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/population_yearly.csv.gz"
)
GEOJSON_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/boundaries/sigungu_kr.geojson"
)

CATEGORY_ORDER = [
    "19% 미만",
    "19% 이상 23% 미만",
    "23% 이상 28% 미만",
    "28% 이상 38% 미만",
    "38% 이상",
]

CATEGORY_COLORS = {
    "19% 미만": "#FFF5F0",
    "19% 이상 23% 미만": "#FDC9B4",
    "23% 이상 28% 미만": "#FC8D59",
    "28% 이상 38% 미만": "#E34A33",
    "38% 이상": "#99000D",
}


def request_file(url: str) -> bytes:
    """인터넷 주소에서 파일을 내려받습니다."""
    headers = {
        "User-Agent": "Mozilla/5.0 Streamlit-App",
        "Accept": "*/*",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.content


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_population() -> pd.DataFrame:
    """
    읍·면·동 인구 자료를 불러옵니다.

    코드 열은 계산할 숫자가 아니라 행정구역 이름표이므로
    반드시 문자열로 읽습니다.
    """
    file_bytes = request_file(POPULATION_URL)

    population = pd.read_csv(
        BytesIO(file_bytes),
        compression="gzip",
        dtype={"코드": "string"},
        low_memory=False,
    )

    population.columns = population.columns.astype(str).str.strip()

    required_columns = {"연도", "시도", "시군구", "동", "코드", "계_0세"}
    missing_columns = required_columns - set(population.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"인구 데이터에 필요한 열이 없습니다: {missing_text}")

    return population


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_geojson() -> dict:
    """전국 시군구 경계 GeoJSON을 불러옵니다."""
    geojson = json.loads(request_file(GEOJSON_URL).decode("utf-8-sig"))

    if "features" not in geojson:
        raise ValueError("GeoJSON에서 features 항목을 찾을 수 없습니다.")

    for feature in geojson["features"]:
        properties = feature.get("properties", {})
        properties["코드"] = str(properties.get("코드", "")).strip().zfill(5)

    return geojson


def clean_number(series: pd.Series) -> pd.Series:
    """
    '1,234'처럼 쉼표가 포함된 인구도 숫자로 바꿉니다.
    읽을 수 없는 값은 0으로 처리합니다.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0)


def calculate_sigungu_birth_ratio(
    population: pd.DataFrame,
    geojson: dict,
) -> tuple[pd.DataFrame, int]:
    """
    가장 최신 연도의 읍·면·동 자료를 시군구 단위로 합칩니다.

    이 자료에는 실제 출생아 수나 합계출산율 열이 없으므로,
    다음 비율을 출산 관련 지표로 사용합니다.

        0세 인구 비율(%) = 계_0세 인구 ÷ 전체 인구 × 100
    """
    data = population.copy()

    data["연도_숫자"] = pd.to_numeric(data["연도"], errors="coerce")
    latest_year = int(data["연도_숫자"].dropna().max())
    data = data[data["연도_숫자"] == latest_year].copy()

    data["코드"] = (
        data["코드"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )

    data["시군구코드"] = data["코드"].str[:5]

    total_age_columns = [
        column for column in data.columns
        if str(column).startswith("계_")
    ]

    if "계_0세" not in total_age_columns:
        raise ValueError("'계_0세' 열을 찾을 수 없습니다.")

    for column in total_age_columns:
        data[column] = clean_number(data[column])

    data["전체인구"] = data[total_age_columns].sum(axis=1)
    data["0세인구"] = data["계_0세"]

    sigungu = (
        data.groupby("시군구코드", as_index=False)
        .agg(
            전체인구=("전체인구", "sum"),
            출생관련인구=("0세인구", "sum"),
        )
    )

    sigungu["출산율"] = np.where(
        sigungu["전체인구"] > 0,
        sigungu["출생관련인구"] / sigungu["전체인구"] * 100,
        np.nan,
    )

    boundary_names = []
    for feature in geojson["features"]:
        properties = feature.get("properties", {})
        boundary_names.append(
            {
                "시군구코드": str(properties.get("코드", "")).zfill(5),
                "시도": properties.get("시도", ""),
                "시군구": properties.get("시군구", ""),
            }
        )

    boundary_df = pd.DataFrame(boundary_names).drop_duplicates("시군구코드")

    result = boundary_df.merge(
        sigungu,
        on="시군구코드",
        how="left",
        validate="one_to_one",
    )

    result["출산율구간"] = pd.cut(
        result["출산율"],
        bins=[-np.inf, 19, 23, 28, 38, np.inf],
        labels=CATEGORY_ORDER,
        right=False,
        ordered=True,
    )

    result["출산율구간"] = pd.Categorical(
        result["출산율구간"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )

    return result, latest_year


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1250px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        .title-box {
            padding: 1.5rem 1.7rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #fff1f2 0%, #ffffff 100%);
            border: 1px solid #fecdd3;
            margin-bottom: 1rem;
        }

        .title-box h1 {
            margin: 0;
            color: #881337;
            font-size: 2.1rem;
        }

        .title-box p {
            margin: 0.55rem 0 0;
            color: #475569;
            line-height: 1.65;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-box">
        <h1>👶 전국 시군구별 출산율 지도</h1>
        <p>
            가장 최신 연도의 읍·면·동 인구를 시군구 코드 앞 5자리로 합산하여
            출산 관련 인구 비율을 단계구분도로 나타냅니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("전국 인구와 지도 경계 데이터를 불러오는 중입니다..."):
        population_df = load_population()
        korea_geojson = load_geojson()
        map_df, selected_year = calculate_sigungu_birth_ratio(
            population_df,
            korea_geojson,
        )

except requests.RequestException as error:
    st.error(
        "인터넷에서 데이터를 내려받지 못했습니다. "
        "잠시 후 새로고침해 주세요."
    )
    st.exception(error)
    st.stop()

except Exception as error:
    st.error("데이터를 처리하는 중 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


st.info(
    f"**기준 연도: {selected_year}년**  \n"
    "이 인구 자료에는 통계청의 합계출산율이나 출생아 수 열이 없습니다. "
    "따라서 이 앱의 출산율은 **0세 인구 ÷ 전체 인구 × 100**으로 계산한 "
    "'0세 인구 비율'입니다."
)

valid_count = int(map_df["출산율"].notna().sum())
missing_count = int(map_df["출산율"].isna().sum())
national_ratio = (
    map_df["출생관련인구"].sum()
    / map_df["전체인구"].sum()
    * 100
)

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("지도 기준 연도", f"{selected_year}년")
metric_2.metric("전국 0세 인구 비율", f"{national_ratio:.2f}%")
metric_3.metric("자료가 연결된 시군구", f"{valid_count}곳")

if missing_count:
    st.warning(
        f"인구 자료와 연결되지 않은 시군구 경계가 {missing_count}곳 있습니다. "
        "행정구역 개편 시점이나 코드 차이 때문일 수 있습니다."
    )

figure = px.choropleth(
    map_df,
    geojson=korea_geojson,
    locations="시군구코드",
    featureidkey="properties.코드",
    color="출산율구간",
    category_orders={"출산율구간": CATEGORY_ORDER},
    color_discrete_map=CATEGORY_COLORS,
    hover_name="시군구",
    hover_data={
        "시군구코드": False,
        "출산율구간": False,
        "시도": True,
        "시군구": False,
        "출산율": ":.2f",
        "전체인구": ":,.0f",
        "출생관련인구": ":,.0f",
    },
    labels={
        "시도": "시도",
        "출산율": "출산율(%)",
        "전체인구": "전체 인구",
        "출생관련인구": "0세 인구",
        "출산율구간": "출산율 구간",
    },
)

figure.update_geos(
    fitbounds="locations",
    visible=False,
    projection_type="mercator",
    bgcolor="rgba(0,0,0,0)",
)

figure.update_traces(
    marker_line_color="#64748B",
    marker_line_width=0.45,
)

figure.update_layout(
    height=760,
    margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    geo_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        title="출산율 구간",
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        traceorder="normal",
    ),
)

st.plotly_chart(
    figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
    },
)

st.subheader("시군구별 출산율 순위")

ranking_df = map_df.dropna(subset=["출산율"]).copy()

high_3 = ranking_df.nlargest(3, "출산율")[
    ["시도", "시군구", "출산율", "출생관련인구", "전체인구"]
].copy()

low_3 = ranking_df.nsmallest(3, "출산율")[
    ["시도", "시군구", "출산율", "출생관련인구", "전체인구"]
].copy()

for table in (high_3, low_3):
    table["출산율"] = table["출산율"].round(2)
    table.rename(
        columns={
            "출산율": "출산율(%)",
            "출생관련인구": "0세 인구(명)",
            "전체인구": "전체 인구(명)",
        },
        inplace=True,
    )

left_column, right_column = st.columns(2)

with left_column:
    st.markdown("#### 🔺 출산율이 높은 지역 3곳")
    st.dataframe(
        high_3,
        hide_index=True,
        use_container_width=True,
        column_config={
            "출산율(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "0세 인구(명)": st.column_config.NumberColumn(format="%d"),
            "전체 인구(명)": st.column_config.NumberColumn(format="%d"),
        },
    )

with right_column:
    st.markdown("#### 🔻 출산율이 낮은 지역 3곳")
    st.dataframe(
        low_3,
        hide_index=True,
        use_container_width=True,
        column_config={
            "출산율(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "0세 인구(명)": st.column_config.NumberColumn(format="%d"),
            "전체 인구(명)": st.column_config.NumberColumn(format="%d"),
        },
    )

with st.expander("자료와 계산 방법 보기"):
    st.markdown(
        f"""
        - **인구 자료:** `{POPULATION_URL}`
        - **지도 경계:** `{GEOJSON_URL}`
        - **기준 연도:** 데이터에 포함된 가장 최신 연도인 **{selected_year}년**
        - **시군구 연결 기준:** 행정동 코드 10자리의 앞 5자리
        - **출산율 계산:** `계_0세 인구 ÷ 전체 연령 인구 × 100`
        - **단계 구간:** 19% 미만 / 19~23% / 23~28% / 28~38% / 38% 이상

        ※ 일반적으로 사용하는 **합계출산율**은 여성 1명이 평생 낳을 것으로
        예상되는 평균 출생아 수이며, 이 인구 자료만으로는 계산할 수 없습니다.
        """
    )
'''

out = Path("/mnt/data/birthrate_main.py")
out.write_text(code, encoding="utf-8")
py_compile.compile(str(out), doraise=True)
print(f"created {out} / {out.stat().st_size:,} bytes / syntax OK")
