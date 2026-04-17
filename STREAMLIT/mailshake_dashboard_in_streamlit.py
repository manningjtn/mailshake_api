import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Daily Campaign Metrics", layout="wide")
st.title("Daily Campaign Metrics")

session = get_active_session()


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(ttl=300)
def get_teams():
    query = """
        SELECT DISTINCT TEAM_ID
        FROM MAILSHAKE_API.MAIL_API.ACTIVITY_SENT_TGT
        WHERE TEAM_ID IS NOT NULL
        ORDER BY TEAM_ID
    """
    return session.sql(query).to_pandas()


@st.cache_data(ttl=300)
def get_campaign_titles(selected_team: str):
    if selected_team == "ALL":
        query = """
            SELECT DISTINCT CAMPAIGN_TITLE
            FROM MAILSHAKE_API.MAIL_API.ACTIVITY_SENT_TGT
            WHERE CAMPAIGN_TITLE IS NOT NULL
            ORDER BY CAMPAIGN_TITLE
        """
    else:
        safe_team = str(selected_team).replace("'", "''")
        query = f"""
            SELECT DISTINCT CAMPAIGN_TITLE
            FROM MAILSHAKE_API.MAIL_API.ACTIVITY_SENT_TGT
            WHERE CAMPAIGN_TITLE IS NOT NULL
              AND TEAM_ID = '{safe_team}'
            ORDER BY CAMPAIGN_TITLE
        """
    return session.sql(query).to_pandas()


def build_where_clause(
    selected_team: str,
    selected_campaign_title: str = "ALL",
    timestamp_column: str = "UPDATE_DTS"
) -> str:
    where_clause = f"WHERE {timestamp_column} >= DATEADD(hour, -24, CURRENT_TIMESTAMP())"

    if selected_team != "ALL":
        safe_team = str(selected_team).replace("'", "''")
        where_clause += f" AND TEAM_ID = '{safe_team}'"

    if selected_campaign_title != "ALL":
        safe_title = str(selected_campaign_title).replace("'", "''")
        where_clause += f" AND CAMPAIGN_TITLE = '{safe_title}'"

    return where_clause


@st.cache_data(ttl=300)
def run_metric_query(
    table_name: str,
    metric_name: str,
    selected_team: str,
    selected_campaign_title: str,
    distinct_email: bool = False,
    timestamp_column: str = "UPDATE_DTS",
):
    where_clause = build_where_clause(
        selected_team,
        selected_campaign_title=selected_campaign_title,
        timestamp_column=timestamp_column
    )
    table_name = f"{table_name}_TGT"

    if distinct_email:
        count_expr = "COUNT(DISTINCT RECIPIENT_EMAILADDRESS)"
    else:
        count_expr = "COUNT(*)"

    query = f"""
    SELECT 
        CAMPAIGN_TITLE,
        CAMPAIGN_ID,
        TEAM_ID,
        {count_expr} AS {metric_name}
    FROM MAILSHAKE_API.MAIL_API.{table_name}
    {where_clause}
    GROUP BY TEAM_ID, CAMPAIGN_TITLE, CAMPAIGN_ID
    """

    df = session.sql(query).to_pandas()

    if df.empty:
        return df

    return df.sort_values(metric_name, ascending=False)


@st.cache_data(ttl=300)
def run_bounce_query(selected_team: str, selected_campaign_title: str):
    where_clause = build_where_clause(
        selected_team,
        selected_campaign_title=selected_campaign_title,
        timestamp_column="UPDATE_DTS"
    )

    query = f"""
    SELECT
        CAMPAIGN_TITLE,
        CAMPAIGN_ID,
        TEAM_ID,
        COUNT(*) AS BOUNCE_COUNT
    FROM MAILSHAKE_API.MAIL_API.ACTIVITY_REPLIES_TGT
    {where_clause}
      AND LOWER(COALESCE(TYPE, '')) = 'bounce'
    GROUP BY TEAM_ID, CAMPAIGN_TITLE, CAMPAIGN_ID
    """

    df = session.sql(query).to_pandas()

    if df.empty:
        return df

    return df.sort_values("BOUNCE_COUNT", ascending=False)


@st.cache_data(ttl=300)
def run_unsubscribe_query(selected_team: str, selected_campaign_title: str):
    where_clause = build_where_clause(
        selected_team,
        selected_campaign_title=selected_campaign_title,
        timestamp_column="UPDATE_DTS"
    )

    query = f"""
    SELECT
        CAMPAIGN_TITLE,
        CAMPAIGN_ID,
        TEAM_ID,
        COUNT(*) AS UNSUBSCRIBE_COUNT
    FROM MAILSHAKE_API.MAIL_API.ACTIVITY_REPLIES_TGT
    {where_clause}
      AND LOWER(COALESCE(TYPE, '')) = 'unsubscribe'
    GROUP BY TEAM_ID, CAMPAIGN_TITLE, CAMPAIGN_ID
    """

    df = session.sql(query).to_pandas()

    if df.empty:
        return df

    return df.sort_values("UNSUBSCRIBE_COUNT", ascending=False)


def make_chart(df: pd.DataFrame, metric_col: str, chart_title: str):
    if df.empty:
        return None

    chart_df = df.head(20)

    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("CAMPAIGN_TITLE:N", sort="-y", title="Campaign Title"),
            y=alt.Y(f"{metric_col}:Q", title=metric_col.replace("_", " ").title()),
            tooltip=["TEAM_ID", "CAMPAIGN_TITLE", "CAMPAIGN_ID", metric_col]
        )
        .properties(title=chart_title, height=400)
    )


def render_metric_tab(df: pd.DataFrame, metric_col: str, chart_title: str):
    total = int(df[metric_col].sum()) if not df.empty else 0
    st.metric(f"Total {metric_col.replace('_', ' ').title()}", f"{total:,}")

    if df.empty:
        st.info("No data found for this selection.")
        return

    st.dataframe(df, use_container_width=True)
    chart = make_chart(df, metric_col, chart_title)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)


def build_combined_team_df(sent_df, opens_df, clicks_df, replies_df, bounces_df, unsubs_df):
    def agg(df, col):
        if df.empty:
            return pd.DataFrame(columns=["TEAM_ID", col])
        return df.groupby("TEAM_ID", as_index=False)[col].sum()

    sent = agg(sent_df, "CAMPAIGN_SENT_MESSAGES")
    opens = agg(opens_df, "OPEN_COUNT")
    clicks = agg(clicks_df, "CLICK_COUNT")
    replies = agg(replies_df, "REPLY_COUNT")
    bounces = agg(bounces_df, "BOUNCE_COUNT")
    unsubs = agg(unsubs_df, "UNSUBSCRIBE_COUNT")

    combined = sent
    for other in [opens, clicks, replies, bounces, unsubs]:
        combined = combined.merge(other, on="TEAM_ID", how="outer")

    combined = combined.fillna(0)

    numeric_cols = [
        "CAMPAIGN_SENT_MESSAGES",
        "OPEN_COUNT",
        "CLICK_COUNT",
        "REPLY_COUNT",
        "BOUNCE_COUNT",
        "UNSUBSCRIBE_COUNT",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = combined[col].astype(int)

    return combined.sort_values("TEAM_ID")


def make_combined_line_chart(combined_df: pd.DataFrame):
    if combined_df.empty:
        return None

    long_df = combined_df.melt(
        id_vars="TEAM_ID",
        value_vars=[
            "CAMPAIGN_SENT_MESSAGES",
            "OPEN_COUNT",
            "CLICK_COUNT",
            "REPLY_COUNT",
            "BOUNCE_COUNT",
            "UNSUBSCRIBE_COUNT",
        ],
        var_name="METRIC",
        value_name="VALUE",
    )

    long_df["METRIC"] = long_df["METRIC"].replace({
        "CAMPAIGN_SENT_MESSAGES": "Sent",
        "OPEN_COUNT": "Opens",
        "CLICK_COUNT": "Clicks",
        "REPLY_COUNT": "Replies",
        "BOUNCE_COUNT": "Bounces",
        "UNSUBSCRIBE_COUNT": "Unsubscribes",
    })

    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("TEAM_ID:N", title="Team ID"),
            y=alt.Y("VALUE:Q", title="Count"),
            color=alt.Color("METRIC:N", title="Metric"),
            tooltip=["TEAM_ID", "METRIC", "VALUE"],
        )
        .properties(title="All Metrics by Team (Last 24h by UPDATE_DTS)", height=450)
    )


# ----------------------------
# Filters
# ----------------------------
teams_df = get_teams()
team_options = ["ALL"] + teams_df["TEAM_ID"].astype(str).tolist()

selected_team = st.selectbox("Select Team", options=team_options)

campaigns_df = get_campaign_titles(selected_team)
campaign_options = ["ALL"] + campaigns_df["CAMPAIGN_TITLE"].dropna().astype(str).tolist()

selected_campaign_title = st.selectbox("Select Campaign Title", options=campaign_options)

title = "All Teams" if selected_team == "ALL" else f"Team {selected_team}"
campaign_label = "All Campaigns" if selected_campaign_title == "ALL" else selected_campaign_title
st.subheader(f"Campaign Metrics (Last 24h by UPDATE_DTS) — {title} — {campaign_label}")


# ----------------------------
# Queries
# ----------------------------
sent_df = run_metric_query(
    table_name="ACTIVITY_SENT",
    metric_name="CAMPAIGN_SENT_MESSAGES",
    selected_team=selected_team,
    selected_campaign_title=selected_campaign_title,
    distinct_email=True,
    timestamp_column="UPDATE_DTS",
)

opens_df = run_metric_query(
    table_name="ACTIVITY_OPENS",
    metric_name="OPEN_COUNT",
    selected_team=selected_team,
    selected_campaign_title=selected_campaign_title,
    timestamp_column="UPDATE_DTS",
)

clicks_df = run_metric_query(
    table_name="ACTIVITY_CLICKS",
    metric_name="CLICK_COUNT",
    selected_team=selected_team,
    selected_campaign_title=selected_campaign_title,
    timestamp_column="UPDATE_DTS",
)

replies_df = run_metric_query(
    table_name="ACTIVITY_REPLIES",
    metric_name="REPLY_COUNT",
    selected_team=selected_team,
    selected_campaign_title=selected_campaign_title,
    timestamp_column="UPDATE_DTS",
)

bounces_df = run_bounce_query(selected_team, selected_campaign_title)
unsubs_df = run_unsubscribe_query(selected_team, selected_campaign_title)

combined_team_df = build_combined_team_df(
    sent_df,
    opens_df,
    clicks_df,
    replies_df,
    bounces_df,
    unsubs_df,
)

combined_line_chart = make_combined_line_chart(combined_team_df)


# ----------------------------
# Top-level KPIs
# ----------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Total Sent",
        f"{int(sent_df['CAMPAIGN_SENT_MESSAGES'].sum()) if not sent_df.empty else 0:,}"
    )

with col2:
    st.metric(
        "Total Opens",
        f"{int(opens_df['OPEN_COUNT'].sum()) if not opens_df.empty else 0:,}"
    )

with col3:
    st.metric(
        "Total Clicks",
        f"{int(clicks_df['CLICK_COUNT'].sum()) if not clicks_df.empty else 0:,}"
    )

with col4:
    st.metric(
        "Total Replies",
        f"{int(replies_df['REPLY_COUNT'].sum()) if not replies_df.empty else 0:,}"
    )

with col5:
    st.metric(
        "Total Bounces",
        f"{int(bounces_df['BOUNCE_COUNT'].sum()) if not bounces_df.empty else 0:,}"
    )

with col6:
    st.metric(
        "Total Unsubscribes",
        f"{int(unsubs_df['UNSUBSCRIBE_COUNT'].sum()) if not unsubs_df.empty else 0:,}"
    )


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Sent", "Opens", "Clicks", "Replies", "Bounces", "Unsubscribes", "All Metrics"]
)

with tab1:
    render_metric_tab(
        sent_df,
        metric_col="CAMPAIGN_SENT_MESSAGES",
        chart_title="Sent by Campaign Title (Last 24h by UPDATE_DTS)"
    )

with tab2:
    render_metric_tab(
        opens_df,
        metric_col="OPEN_COUNT",
        chart_title="Opens by Campaign Title (Last 24h by UPDATE_DTS)"
    )

with tab3:
    render_metric_tab(
        clicks_df,
        metric_col="CLICK_COUNT",
        chart_title="Clicks by Campaign Title (Last 24h by UPDATE_DTS)"
    )

with tab4:
    render_metric_tab(
        replies_df,
        metric_col="REPLY_COUNT",
        chart_title="Replies by Campaign Title (Last 24h by UPDATE_DTS)"
    )

with tab5:
    render_metric_tab(
        bounces_df,
        metric_col="BOUNCE_COUNT",
        chart_title="Bounces by Campaign Title (Last 24h by UPDATE_DTS)"
    )

with tab6:
    render_metric_tab(
        unsubs_df,
        metric_col="UNSUBSCRIBE_COUNT",
        chart_title="Unsubscribes by Campaign Title (Last 24h by UPDATE_DTS)"
    )

with tab7:
    if combined_team_df.empty:
        st.info("No data found for this selection.")
    else:
        st.metric(
            "Teams in Combined View",
            f"{combined_team_df['TEAM_ID'].nunique():,}"
        )
        st.dataframe(combined_team_df, use_container_width=True)

        if combined_line_chart is not None:
            st.altair_chart(combined_line_chart, use_container_width=True)
