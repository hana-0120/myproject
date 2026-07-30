from pathlib import Path
import py_compile

app = '''import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# 1. 앱 기본 설정
# =========================================================
st.set_page_config(
    page_title="전국 시군구 고령화 지도",
    page_icon="🗺️",
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


# =========================================================
# 2. 화면 꾸미기
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .title-box {
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border: 1px solid #fed7aa;
        margin-bottom: 1rem;
    }

    .title-box h1 {
        margin: 0;
        font-size: 2.15rem;
    }

    .title-box p {
        margin: 0.5rem 0 0 0;
        color: #57534e;
        line-height: 1.6;
    }

    .guide {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. 데이터 내려받기
# =========================================================
@st.cache_data(ttl=43200, show_spinner=False)
def load_population(url):
    """압축된 CSV를 내려받습니다. 코드는 반드시 문자열로 읽습니다."""

    response = requests.get(url, timeout=90)
    response.raise_for_status()

    return pd.read_csv(
        io.BytesIO(response.content),
        compression="gzip",
        dtype={"코드": "string"},
        low_memory=False,
    )


@st.cache_data(ttl=43200, show_spinner=False)
def load_geojson(url):
    """전국 시군구 경계 GeoJSON을 내려받습니다."""

    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


# =========================================================
# 4. 보조 함수
# =========================================================
def to_number(series):
    """쉼표나 공백이 들어간 인구 값을 숫자로 바꿉니다."""

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def normalize_code(series):
    """행정동 코드를 10자리 문자열로 정리합니다."""

    return (
        series.astype("string")
        .str.replace(r"\\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(10)
    )


def age_from_column(column_name):
    """'계_65세', '계_100세 이상'에서 나이 숫자를 읽습니다."""

    text = str(column_name).replace("계_", "", 1).strip()

    if text == "100세 이상":
        return 100

    matched = re.fullmatch(r"(\\d+)세", text)

    if matched:
        return int(matched.group(1))

    return None


# =========================================================
# 5. 최신 연도 시군구별 고령화율 계산
# =========================================================
def make_sigungu_population(population):
    required = {"연도", "코드"}

    if not required.issubset(population.columns):
        missing = required - set(population.columns)
        raise ValueError(
            "인구 데이터에서 필요한 열을 찾지 못했습니다: "
            + ", ".join(sorted(missing))
        )

    data = population.copy()

    data["연도"] = pd.to_numeric(data["연도"], errors="coerce")
    data = data.dropna(subset=["연도"])

    if data.empty:
        raise ValueError("유효한 연도 자료가 없습니다.")

    latest_year = int(data["연도"].max())
    data = data[data["연도"] == latest_year].copy()

    data["코드"] = normalize_code(data["코드"])
    data["시군구코드"] = data["코드"].str[:5]

    total_columns = [
        column
        for column in data.columns
        if str(column).startswith("계_")
        and age_from_column(column) is not None
    ]

    elderly_columns = [
        column
        for column in total_columns
        if age_from_column(column) >= 65
    ]

    if not total_columns:
        raise ValueError("'계_0세' 형식의 인구 열을 찾지 못했습니다.")

    if not elderly_columns:
        raise ValueError("65세 이상 인구 열을 찾지 못했습니다.")

    for column in total_columns:
        data[column] = to_number(data[column])

    data["전체인구"] = data[total_columns].sum(axis=1)
    data["고령인구"] = data[elderly_columns].sum(axis=1)

    grouped = (
        data.groupby("시군구코드", as_index=False)
        .agg(
            전체인구=("전체인구", "sum"),
            고령인구=("고령인구", "sum"),
        )
    )

    grouped = grouped[grouped["전체인구"] > 0].copy()
    grouped["고령화율"] = (
        grouped["고령인구"]
        / grouped["전체인구"]
        * 100
    )

    return grouped, latest_year


# =========================================================
# 6. GeoJSON 속성을 표로 만들기
# =========================================================
def make_boundary_table(geojson):
    rows = []

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})

        code = str(properties.get("코드", "")).strip()
        code = re.sub(r"[^0-9]", "", code).zfill(5)

        sido = str(properties.get("시도", "")).strip()
        sigungu = str(properties.get("시군구", "")).strip()

        feature["properties"]["코드"] = code

        if code:
            rows.append(
                {
                    "시군구코드": code,
                    "시도": sido,
                    "시군구": sigungu,
                }
            )

    table = pd.DataFrame(rows)

    if table.empty:
        raise ValueError("GeoJSON에서 시군구 정보를 읽지 못했습니다.")

    return table.drop_duplicates("시군구코드")


# =========================================================
# 7. 고령화율을 5단계로 구분
# =========================================================
def add_groups(data):
    result = data.copy()

    conditions = [
        result["고령화율"] < 19,
        (result["고령화율"] >= 19)
        & (result["고령화율"] < 23),
        (result["고령화율"] >= 23)
        & (result["고령화율"] < 28),
        (result["고령화율"] >= 28)
        & (result["고령화율"] < 38),
        result["고령화율"] >= 38,
    ]

    result["단계"] = np.select(
        conditions,
        [0, 1, 2, 3, 4],
        default=np.nan,
    )

    labels = {
        0: "19% 미만",
        1: "19% 이상 23% 미만",
        2: "23% 이상 28% 미만",
        3: "28% 이상 38% 미만",
        4: "38% 이상",
    }

    result["비율구간"] = result["단계"].map(labels)
    return result


# =========================================================
# 8. Plotly 단계구분도 만들기
# =========================================================
def make_map(data, geojson):
    colors = [
        "#fff7ec",
        "#fee8c8",
        "#fdbb84",
        "#e34a33",
        "#7f0000",
    ]

    color_scale = [
        [0.0000, colors[0]],
        [0.1999, colors[0]],
        [0.2000, colors[1]],
        [0.3999, colors[1]],
        [0.4000, colors[2]],
        [0.5999, colors[2]],
        [0.6000, colors[3]],
        [0.7999, colors[3]],
        [0.8000, colors[4]],
        [1.0000, colors[4]],
    ]

    custom_data = np.stack(
        [
            data["시군구"],
            data["시도"],
            data["고령화율"],
            data["비율구간"],
        ],
        axis=-1,
    )

    figure = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=data["시군구코드"],
            featureidkey="properties.코드",
            z=data["단계"],
            zmin=0,
            zmax=4,
            colorscale=color_scale,
            customdata=custom_data,
            marker_line_color="white",
            marker_line_width=0.45,
            hovertemplate=(
                "<b>%{customdata[1]} %{customdata[0]}</b><br>"
                "고령화율: %{customdata[2]:.1f}%<br>"
                "구간: %{customdata[3]}"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="고령화율",
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=[
                    "19% 미만",
                    "19% 이상 23% 미만",
                    "23% 이상 28% 미만",
                    "28% 이상 38% 미만",
                    "38% 이상",
                ],
                thickness=18,
                len=0.75,
                x=1.01,
                outlinewidth=0,
            ),
        )
    )

    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )

    figure.update_layout(
        height=760,
        margin=dict(l=0, r=135, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return figure


# =========================================================
# 9. 표에 사용할 자료 정리
# =========================================================
def ranking_table(data, highest=True):
    columns = [
        "시도",
        "시군구",
        "고령화율",
        "전체인구",
        "고령인구",
    ]

    if highest:
        table = data.nlargest(3, "고령화율")[columns]
    else:
        table = data.nsmallest(3, "고령화율")[columns]

    table = table.copy().reset_index(drop=True)
    table.index = table.index + 1

    table["고령화율"] = table["고령화율"].map(
        lambda value: f"{value:.1f}%"
    )
    table["전체인구"] = table["전체인구"].map(
        lambda value: f"{int(value):,}명"
    )
    table["고령인구"] = table["고령인구"].map(
        lambda value: f"{int(value):,}명"
    )

    return table


# =========================================================
# 10. 실제 화면 출력
# =========================================================
st.markdown(
    """
    <div class="title-box">
        <h1>🗺️ 전국 시군구 고령화 지도</h1>
        <p>
            전국 읍·면·동 인구를 시군구 단위로 합쳐
            65세 이상 인구 비율을 5단계 색으로 나타냅니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="guide">
        <b>계산 방법</b><br>
        고령화율 = 65세 이상 인구 ÷ 전체 인구 × 100<br>
        지도와 인구 자료는 지역 이름이 아니라
        <b>행정동 코드 앞 5자리</b>로 연결합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("최신 인구 자료와 지도 경계를 불러오는 중입니다."):
        population = load_population(POPULATION_URL)
        geojson = load_geojson(GEOJSON_URL)

    sigungu_population, latest_year = make_sigungu_population(
        population
    )
    boundary = make_boundary_table(geojson)

    map_data = boundary.merge(
        sigungu_population,
        on="시군구코드",
        how="left",
        validate="one_to_one",
    )

    unmatched_count = int(map_data["고령화율"].isna().sum())

    map_data = map_data.dropna(
        subset=["고령화율"]
    ).copy()

    if map_data.empty:
        raise ValueError(
            "인구 자료와 지도 경계가 코드로 연결되지 않았습니다."
        )

    map_data = add_groups(map_data)

    national_rate = (
        map_data["고령인구"].sum()
        / map_data["전체인구"].sum()
        * 100
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("기준 연도", f"{latest_year}년")
    metric2.metric("지도 표시 지역", f"{len(map_data):,}개")
    metric3.metric(
        "전국 고령화율",
        f"{national_rate:.1f}%",
    )

    figure = make_map(map_data, geojson)

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
        },
    )

    st.caption(
        "색이 진할수록 65세 이상 인구 비율이 높습니다. "
        "지역 위에 마우스를 올리면 상세 값을 볼 수 있습니다."
    )

    if unmatched_count > 0:
        st.warning(
            f"지도 경계 중 인구 자료와 연결되지 않은 지역이 "
            f"{unmatched_count}개 있습니다. "
            "행정구역 개편이나 코드 차이일 수 있습니다."
        )

    high_column, low_column = st.columns(2)

    with high_column:
        st.subheader("🔴 고령화율이 높은 지역 3곳")
        st.dataframe(
            ranking_table(map_data, highest=True),
            use_container_width=True,
        )

    with low_column:
        st.subheader("🟢 고령화율이 낮은 지역 3곳")
        st.dataframe(
            ranking_table(map_data, highest=False),
            use_container_width=True,
        )

    with st.expander("자료와 처리 방법 보기"):
        st.write(f"- 인구 자료: {POPULATION_URL}")
        st.write(f"- 지도 경계: {GEOJSON_URL}")
        st.write(f"- 최신 연도 자동 선택: {latest_year}년")
        st.write("- 전체 인구: 모든 '계_나이' 열의 합")
        st.write("- 고령 인구: 65세부터 100세 이상까지의 합")
        st.write("- 시군구 연결 기준: 행정동 코드 앞 5자리")

except requests.RequestException as error:
    st.error(
        "인터넷에서 자료를 내려받지 못했습니다. "
        "잠시 후 새로고침해 주세요."
    )
    st.exception(error)

except Exception as error:
    st.error(
        "자료를 처리하는 중 오류가 발생했습니다. "
        "아래 상세 내용을 확인해 주세요."
    )
    st.exception(error)
'''

path = Path("/mnt/data/main_500.py")
path.write_text(app, encoding="utf-8")
py_compile.compile(str(path), doraise=True)

text = path.read_text(encoding="utf-8")
print("생성 파일:", path)
print("전체 줄 수:", len(text.splitlines()))
print("실행 코드 안 write_text 포함:", "write_text" in text)
print("실행 코드 안 pathlib 포함:", "pathlib" in text.lower())
print("문법 검사: 통과")
print(f"created {out} / {out.stat().st_size:,} bytes / syntax OK")
