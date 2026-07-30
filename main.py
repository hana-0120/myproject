from pathlib import Path

main_code = r'''import io
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
    page_title="전국 시군구 출산 관련 지표 지도",
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

# 단계 구간 경계값
BREAKS = [19, 23, 28, 38]

# 단계별 색상: 낮은 값은 옅게, 높은 값은 진하게 표시
COLORS = [
    "#fff5f0",
    "#fee0d2",
    "#fcbba1",
    "#fb6a4a",
    "#a50f15",
]

LABELS = [
    "19% 미만",
    "19% 이상 23% 미만",
    "23% 이상 28% 미만",
    "28% 이상 38% 미만",
    "38% 이상",
]


# =========================================================
# 2. 화면 디자인
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 1px solid #fecdd3;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        color: #881337;
    }

    .hero p {
        margin: 0.6rem 0 0 0;
        color: #57534e;
        line-height: 1.65;
    }

    .notice {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. 원격 데이터 불러오기
# =========================================================
@st.cache_data(ttl=43200, show_spinner=False)
def load_population(url):
    """
    압축된 CSV 인구 자료를 내려받습니다.

    '코드'는 계산용 숫자가 아니라 행정구역 이름표이므로
    반드시 문자열로 읽습니다.
    """
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
# 4. 자료 정리 보조 함수
# =========================================================
def clean_number(series):
    """
    인구 값에 쉼표나 공백이 들어 있어도
    계산할 수 있도록 숫자로 바꿉니다.
    """
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    ).fillna(0)


def normalize_admin_code(series):
    """
    행정동 코드를 10자리 문자열로 맞춥니다.

    CSV에서 코드가 1234567890.0처럼 읽힌 경우도 처리합니다.
    """
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(10)
    )


def find_age_column(columns, prefix, age):
    """
    열 목록에서 '계_0세', '여_15세' 같은 나이별 열을 찾습니다.

    데이터 열 이름에 불필요한 공백이 있어도 찾을 수 있게 합니다.
    """
    wanted = f"{prefix}_{age}세"

    for column in columns:
        if str(column).strip() == wanted:
            return column

    return None


# =========================================================
# 5. 시군구별 출산 관련 지표 계산
# =========================================================
def prepare_fertility_proxy(population):
    """
    최신 연도의 읍면동 인구를 시군구별로 합칩니다.

    이 자료에는 실제 출생아 수나 합계출산율 열이 없으므로,
    국제적으로 인구 자료에서 사용하는 출산 관련 대체 지표인
    '아동-가임여성비'를 계산합니다.

    계산식:
    0~4세 전체 인구 ÷ 15~49세 여성 인구 × 100
    """
    required_columns = {"연도", "코드"}
    missing = required_columns - set(population.columns)

    if missing:
        raise ValueError(
            "인구 데이터에 필요한 열이 없습니다: "
            + ", ".join(sorted(missing))
        )

    data = population.copy()

    # 연도를 숫자로 바꾸고 최신 연도를 자동으로 찾습니다.
    data["연도"] = pd.to_numeric(
        data["연도"],
        errors="coerce",
    )
    data = data.dropna(subset=["연도"])

    if data.empty:
        raise ValueError("유효한 연도 자료를 찾지 못했습니다.")

    latest_year = int(data["연도"].max())
    data = data.loc[data["연도"] == latest_year].copy()

    # 행정동 코드 앞 5자리를 시군구 코드로 사용합니다.
    data["코드"] = normalize_admin_code(data["코드"])
    data["시군구코드"] = data["코드"].str[:5]

    # 0~4세 전체 인구 열을 찾습니다.
    child_columns = []

    for age in range(0, 5):
        column = find_age_column(
            data.columns,
            "계",
            age,
        )

        if column is None:
            raise ValueError(
                f"'계_{age}세' 열을 찾지 못했습니다."
            )

        child_columns.append(column)

    # 15~49세 여성 인구 열을 찾습니다.
    women_columns = []

    for age in range(15, 50):
        column = find_age_column(
            data.columns,
            "여",
            age,
        )

        if column is None:
            raise ValueError(
                f"'여_{age}세' 열을 찾지 못했습니다."
            )

        women_columns.append(column)

    # 계산에 필요한 열만 숫자로 바꿉니다.
    needed_columns = child_columns + women_columns

    for column in needed_columns:
        data[column] = clean_number(data[column])

    # 읍면동별 아동 수와 가임여성 수를 계산합니다.
    data["0_4세인구"] = data[child_columns].sum(axis=1)
    data["15_49세여성"] = data[women_columns].sum(axis=1)

    # 읍면동 자료를 시군구 코드로 합칩니다.
    sigungu = (
        data.groupby(
            "시군구코드",
            as_index=False,
        )
        .agg(
            아동인구=("0_4세인구", "sum"),
            가임여성인구=("15_49세여성", "sum"),
        )
    )

    # 분모가 0인 지역은 계산할 수 없으므로 제외합니다.
    sigungu = sigungu.loc[
        sigungu["가임여성인구"] > 0
    ].copy()

    sigungu["출산관련지표"] = (
        sigungu["아동인구"]
        / sigungu["가임여성인구"]
        * 100
    )

    return sigungu, latest_year


# =========================================================
# 6. GeoJSON 속성을 표로 만들기
# =========================================================
def make_boundary_table(geojson):
    """
    GeoJSON의 코드, 시도, 시군구를 데이터프레임으로 만듭니다.

    이름이 같은 '남구' 등이 있으므로
    이름이 아니라 5자리 코드로 연결합니다.
    """
    rows = []

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})

        raw_code = str(
            properties.get("코드", "")
        ).strip()

        code = re.sub(
            r"[^0-9]",
            "",
            raw_code,
        ).zfill(5)

        sido = str(
            properties.get("시도", "")
        ).strip()

        sigungu = str(
            properties.get("시군구", "")
        ).strip()

        # Plotly가 코드를 찾을 수 있도록 문자열로 통일합니다.
        feature.setdefault(
            "properties",
            {},
        )["코드"] = code

        if code:
            rows.append(
                {
                    "시군구코드": code,
                    "시도": sido,
                    "시군구": sigungu,
                }
            )

    boundary = pd.DataFrame(rows)

    if boundary.empty:
        raise ValueError(
            "GeoJSON에서 시군구 속성을 읽지 못했습니다."
        )

    return boundary.drop_duplicates(
        "시군구코드"
    )


# =========================================================
# 7. 값을 5단계 구간으로 나누기
# =========================================================
def add_category(data):
    """출산 관련 지표를 지정된 5개 구간으로 나눕니다."""
    result = data.copy()

    conditions = [
        result["출산관련지표"] < BREAKS[0],
        (
            result["출산관련지표"] >= BREAKS[0]
        )
        & (
            result["출산관련지표"] < BREAKS[1]
        ),
        (
            result["출산관련지표"] >= BREAKS[1]
        )
        & (
            result["출산관련지표"] < BREAKS[2]
        ),
        (
            result["출산관련지표"] >= BREAKS[2]
        )
        & (
            result["출산관련지표"] < BREAKS[3]
        ),
        result["출산관련지표"] >= BREAKS[3],
    ]

    result["단계"] = np.select(
        conditions,
        [0, 1, 2, 3, 4],
        default=np.nan,
    )

    label_map = {
        0: LABELS[0],
        1: LABELS[1],
        2: LABELS[2],
        3: LABELS[3],
        4: LABELS[4],
    }

    result["비율구간"] = result["단계"].map(
        label_map
    )

    return result


# =========================================================
# 8. 배경 타일 없는 단계구분도 만들기
# =========================================================
def make_choropleth(data, geojson):
    """배경 지도 타일 없이 시군구 경계만 표시합니다."""

    # 연속 그라데이션처럼 보이지 않도록
    # 각 단계의 시작과 끝에 같은 색을 반복합니다.
    discrete_scale = [
        [0.0000, COLORS[0]],
        [0.1999, COLORS[0]],
        [0.2000, COLORS[1]],
        [0.3999, COLORS[1]],
        [0.4000, COLORS[2]],
        [0.5999, COLORS[2]],
        [0.6000, COLORS[3]],
        [0.7999, COLORS[3]],
        [0.8000, COLORS[4]],
        [1.0000, COLORS[4]],
    ]

    custom_data = np.stack(
        [
            data["시군구"],
            data["시도"],
            data["출산관련지표"],
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
            colorscale=discrete_scale,
            customdata=custom_data,
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=(
                "<b>%{customdata[1]} %{customdata[0]}</b><br>"
                "출산 관련 지표: %{customdata[2]:.1f}%<br>"
                "구간: %{customdata[3]}"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="비율 구간",
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=LABELS,
                thickness=18,
                len=0.76,
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
        margin=dict(
            l=0,
            r=145,
            t=10,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return figure


# =========================================================
# 9. 상위·하위 지역 표 만들기
# =========================================================
def make_ranking_table(data, highest=True):
    """출산 관련 지표 상위 또는 하위 3개 지역을 정리합니다."""
    columns = [
        "시도",
        "시군구",
        "출산관련지표",
        "아동인구",
        "가임여성인구",
    ]

    if highest:
        table = data.nlargest(
            3,
            "출산관련지표",
        )[columns]
    else:
        table = data.nsmallest(
            3,
            "출산관련지표",
        )[columns]

    table = table.copy().reset_index(
        drop=True
    )
    table.index = table.index + 1

    table = table.rename(
        columns={
            "출산관련지표": "출산 관련 지표",
            "아동인구": "0~4세 인구",
            "가임여성인구": "15~49세 여성",
        }
    )

    table["출산 관련 지표"] = table[
        "출산 관련 지표"
    ].map(
        lambda value: f"{value:.1f}%"
    )

    table["0~4세 인구"] = table[
        "0~4세 인구"
    ].map(
        lambda value: f"{int(value):,}명"
    )

    table["15~49세 여성"] = table[
        "15~49세 여성"
    ].map(
        lambda value: f"{int(value):,}명"
    )

    return table


# =========================================================
# 10. 앱 화면 출력
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>👶 전국 시군구 출산 관련 지표 지도</h1>
        <p>
            최신 연도의 읍·면·동 인구를 시군구 단위로 합쳐
            0~4세 인구와 15~49세 여성 인구의 비율을 표시합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <b>지표 안내</b><br>
        제공된 자료에는 실제 출생아 수나 합계출산율이 없으므로,
        이 앱은 <b>0~4세 인구 ÷ 15~49세 여성 인구 × 100</b>으로
        계산한 ‘아동-가임여성비’를 출산 관련 지표로 사용합니다.
        공식 합계출산율과는 다른 값입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner(
        "최신 인구 자료와 시군구 경계를 불러오는 중입니다."
    ):
        population = load_population(
            POPULATION_URL
        )
        geojson = load_geojson(
            GEOJSON_URL
        )

    fertility_data, latest_year = prepare_fertility_proxy(
        population
    )

    boundary_data = make_boundary_table(
        geojson
    )

    # 지도 경계 255개를 기준으로 코드로 연결합니다.
    map_data = boundary_data.merge(
        fertility_data,
        on="시군구코드",
        how="left",
        validate="one_to_one",
    )

    unmatched_count = int(
        map_data["출산관련지표"].isna().sum()
    )

    # 실제 계산값이 있는 지역만 지도와 순위에 사용합니다.
    map_data = map_data.dropna(
        subset=["출산관련지표"]
    ).copy()

    if map_data.empty:
        raise ValueError(
            "인구 자료와 지도 경계를 코드로 연결하지 못했습니다."
        )

    map_data = add_category(
        map_data
    )

    national_ratio = (
        map_data["아동인구"].sum()
        / map_data["가임여성인구"].sum()
        * 100
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "기준 연도",
        f"{latest_year}년",
    )

    metric2.metric(
        "지도 표시 지역",
        f"{len(map_data):,}개",
    )

    metric3.metric(
        "전국 출산 관련 지표",
        f"{national_ratio:.1f}%",
    )

    figure = make_choropleth(
        map_data,
        geojson,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
        },
    )

    st.caption(
        "색이 진할수록 0~4세 인구 대비 15~49세 여성 인구의 "
        "비율이 높습니다. 지역 위에 마우스를 올리면 상세 값을 볼 수 있습니다."
    )

    if unmatched_count > 0:
        st.warning(
            f"지도 경계 중 인구 자료와 연결되지 않은 지역이 "
            f"{unmatched_count}개 있습니다. "
            "행정구역 개편이나 코드 차이가 원인일 수 있습니다."
        )

    high_column, low_column = st.columns(2)

    with high_column:
        st.subheader(
            "🔴 출산 관련 지표가 높은 지역 3곳"
        )

        st.dataframe(
            make_ranking_table(
                map_data,
                highest=True,
            ),
            use_container_width=True,
        )

    with low_column:
        st.subheader(
            "🔵 출산 관련 지표가 낮은 지역 3곳"
        )

        st.dataframe(
            make_ranking_table(
                map_data,
                highest=False,
            ),
            use_container_width=True,
        )

    with st.expander(
        "자료와 계산 방법 보기"
    ):
        st.write(
            f"- 인구 자료: {POPULATION_URL}"
        )
        st.write(
            f"- 지도 경계: {GEOJSON_URL}"
        )
        st.write(
            f"- 자동 선택된 최신 연도: {latest_year}년"
        )
        st.write(
            "- 분자: 계_0세부터 계_4세까지의 합"
        )
        st.write(
            "- 분모: 여_15세부터 여_49세까지의 합"
        )
        st.write(
            "- 시군구 연결: 10자리 행정동 코드의 앞 5자리"
        )
        st.write(
            "- 공식 합계출산율이 아닌 인구구조 기반 대체 지표"
        )

except requests.RequestException as error:
    st.error(
        "인터넷에서 자료를 내려받지 못했습니다. "
        "잠시 후 다시 실행해 주세요."
    )
    st.exception(error)

except Exception as error:
    st.error(
        "자료를 처리하는 중 오류가 발생했습니다. "
        "아래 상세 내용을 확인해 주세요."
    )
    st.exception(error)
'''

# 문법 검사: 파일 생성 코드 없이 앱 코드 자체만 검사합니다.
compile(main_code, "main.py", "exec")

output_file = Path("/mnt/data/main.py")
output_file.write_text(main_code, encoding="utf-8")

saved = output_file.read_text(encoding="utf-8")

print("파일 생성 완료:", output_file)
print("전체 줄 수:", len(saved.splitlines()))
print("앱 내부 write_text 포함:", "write_text" in saved)
print("앱 내부 pathlib 포함:", "pathlib" in saved.lower())
print("문법 검사: 통과")
