import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="DataLens | Visual EDA", page_icon="◈", layout="wide")
PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#38bdf8", "#a78bfa"]
st.markdown("""<style>
.block-container{max-width:1500px;padding-top:2rem}
[data-testid="stMetric"]{background:#141e32;border:1px solid #29354a;border-radius:16px;padding:18px}
[data-testid="stMetricValue"]{color:#a5b4fc}
h1,h2,h3{letter-spacing:-.035em}
.hero{padding:28px 32px;border:1px solid #364365;border-radius:22px;background:linear-gradient(120deg,#192b46,#25234a);margin-bottom:20px}
.hero h1{margin:0;font-size:3rem;color:#fff}.hero p{color:#b9c9e6;margin:8px 0 0}
.eyebrow{font-size:12px;letter-spacing:3px;color:#a5b4fc;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_data
def read_csv(raw, encoding, delimiter):
    df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=delimiter,
                     engine="python" if delimiter is None else "c")
    return df

def demo():
    rng = np.random.default_rng(42)
    n = 850
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Category": rng.choice(["Electronics", "Home", "Fashion", "Sports"], n),
        "Units": rng.integers(1, 30, n), "Unit price": rng.uniform(200, 2500, n).round(2),
        "Rating": rng.uniform(2.5, 5, n).round(1)})
    df["Revenue"] = (df["Units"] * df["Unit price"]).round(2)
    df.loc[rng.choice(n, 20, replace=False), "Rating"] = np.nan
    return df

def chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#141e32",
        plot_bgcolor="#141e32", font=dict(color="#dbeafe"), margin=dict(l=25,r=25,t=50,b=30),
        title_font_size=18, legend_title_text="", height=390)
    st.plotly_chart(fig, width="stretch", config={"displaylogo":False})

def clean_display(s):
    return s.replace([np.inf, -np.inf], np.nan) if pd.api.types.is_numeric_dtype(s) else s

def profile(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean()*100).round(2).values,
        "Unique": df.nunique().values,
    })

st.markdown('<div class="hero"><div class="eyebrow">EXPLORATORY DATA ANALYSIS / WORKSPACE</div><h1>◈ DataLens</h1><p>From raw CSV to interactive data intelligence.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("01 / Data source")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    encoding = st.selectbox("Encoding", ["utf-8-sig","latin-1","cp1252","utf-16"])
    sep = st.selectbox("Delimiter", ["Auto", "Comma", "Semicolon", "Tab", "Pipe"])
    delimiter = {"Auto":None,"Comma":",","Semicolon":";","Tab":"\t","Pipe":"|"}[sep]
try:
    original = read_csv(upload.getvalue(), encoding, delimiter) if upload else demo()
except Exception as e:
    st.error(f"CSV could not be parsed: {e}. Try another delimiter or encoding.")
    st.stop()
if original.empty:
    st.warning("The dataset has no rows to analyze.")
    st.stop()
original = original.copy()
original.columns = original.columns.astype(str)
with st.sidebar:
    st.caption(upload.name if upload else "Demo dataset • sales transactions")
    candidates = [c for c in original if not pd.api.types.is_numeric_dtype(original[c])]
    date_cols = st.multiselect("Interpret columns as dates", candidates,
        default=["Date"] if "Date" in candidates else [])
    for c in date_cols:
        original[c] = pd.to_datetime(original[c], errors="coerce")
    st.header("02 / Filter")
    categories = [c for c in original if not pd.api.types.is_numeric_dtype(original[c]) and not pd.api.types.is_datetime64_any_dtype(original[c]) and original[c].nunique() <= 100]
    df = original.copy()
    for c in st.multiselect("Categorical filters", categories):
        options = original[c].dropna().unique().tolist()
        selected = st.multiselect(c, options, default=options, key=f"filter_{c}")
        df = df[df[c].isin(selected)]
    num_options = original.select_dtypes(include="number").columns.tolist()
    for c in st.multiselect("Numeric filters", num_options):
        finite = clean_display(original[c]).dropna()
        if not finite.empty:
            lo, hi = float(finite.min()), float(finite.max())
            if lo < hi:
                bounds = st.slider(c, lo, hi, (lo, hi), key=f"range_{c}")
                df = df[df[c].between(*bounds)]
    st.caption(f"{len(df):,} of {len(original):,} rows")
if df.empty:
    st.warning("No rows match these filters.")
    st.stop()
numeric = df.select_dtypes(include="number").columns.tolist()
dates = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
cats = [c for c in df if c not in numeric and c not in dates]
p = profile(df)
missing = int(df.isna().sum().sum())
metrics = st.columns(5)
for col, label, value in zip(metrics, ["Records","Features","Numeric columns","Missing cells","Duplicate rows"], [f"{len(df):,}",df.shape[1],len(numeric),f"{missing:,}",f"{df.duplicated().sum():,}"]):
    col.metric(label,value)
tabs = st.tabs(["Overview","Data quality","Distributions","Relationships","Time series","Chart studio","Data explorer"])
with tabs[0]:
    st.subheader("Dataset overview")
    a,b = st.columns(2)
    with a:
        types = p["Type"].value_counts().rename_axis("Type").reset_index(name="Columns")
        chart(px.pie(types,names="Type",values="Columns",hole=.68,title="Feature types",color_discrete_sequence=PALETTE))
    with b:
        top = p.sort_values("Unique",ascending=False).head(20)
        chart(px.bar(top,x="Unique",y="Column",orientation="h",title="Column cardinality • top 20",color_discrete_sequence=PALETTE))
    st.subheader("Automatic observations")
    st.write(f"**Completeness:** {100*(1-missing/df.size):.1f}% of cells are non-missing.")
    st.write(f"**Duplicates:** {df.duplicated().sum():,} repeated rows (excluding first occurrence).")
    constants = p.loc[p["Unique"]<=1,"Column"].tolist()
    st.write("**Constant or empty features:** "+(", ".join(constants) if constants else "None detected"))
    if numeric:
        desc = df[numeric].replace([np.inf,-np.inf],np.nan).describe().T
        st.dataframe(desc,width="stretch")
with tabs[1]:
    st.subheader("Completeness and integrity")
    top = p.sort_values("Missing %",ascending=False).head(30)
    chart(px.bar(top,x="Column",y="Missing %",color="Missing %",color_continuous_scale="Purples",title="Missing values by feature • top 30"))
    st.dataframe(p,width="stretch",hide_index=True)
    st.download_button("Download quality report",p.to_csv(index=False),"quality_report.csv","text/csv")
    sample = df.iloc[:500,:40].isna().astype(int)
    chart(px.imshow(sample,aspect="auto",color_continuous_scale=[[0,"#24324c"],[1,"#ec4899"]],title="Missingness matrix • first 500 rows / 40 columns",labels={"color":"Missing"}))
with tabs[2]:
    st.subheader("Explore distributions")
    if numeric:
        c = st.selectbox("Numeric feature",numeric)
        s = clean_display(df[c]).dropna()
        if not s.empty:
            a,b = st.columns(2)
            with a:
                bins=st.slider("Histogram bins",10,100,35)
                chart(px.histogram(s.to_frame(),x=c,nbins=bins,marginal="rug",title=f"Distribution · {c}",color_discrete_sequence=PALETTE))
            with b:
                chart(px.box(s.to_frame(),y=c,points="outliers",title=f"Outliers · {c}",color_discrete_sequence=PALETTE))
            q1,q3=s.quantile([.25,.75]); iqr=q3-q1
            st.caption(f"IQR rule flags {((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum():,} potential outliers; these are not necessarily errors.")
    if cats:
        c = st.selectbox("Categorical feature",cats)
        counts = df[c].astype("string").fillna("(Missing)").value_counts().head(25).rename_axis("Category").reset_index(name="Count")
        chart(px.bar(counts,x="Category",y="Count",title="Category frequency • top 25",color_discrete_sequence=["#14b8a6"]))
    if not numeric and not cats: st.info("No numeric or categorical features available.")
with tabs[3]:
    st.subheader("Relationships and correlations")
    if len(numeric)>=2:
        selected = st.multiselect("Correlation features (max 30)",numeric,default=numeric[:min(12,len(numeric))],max_selections=30)
        method=st.radio("Correlation method",["pearson","spearman"],horizontal=True)
        if len(selected)>=2:
            corr=df[selected].replace([np.inf,-np.inf],np.nan).corr(method=method)
            chart(px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale="RdBu_r",text_auto=".2f",title=f"{method.title()} correlation"))
        a,b=st.columns(2)
        x=a.selectbox("X axis",numeric); y=b.selectbox("Y axis",numeric,index=1)
        hue=st.selectbox("Color group",[None]+[c for c in cats if df[c].nunique()<=20])
        sample=df.sample(min(len(df),10000),random_state=42)
        sample=sample.replace([np.inf,-np.inf],np.nan)
        chart(px.scatter(sample,x=x,y=y,color=hue,opacity=.65,title="Scatter plot • up to 10,000 rows",color_discrete_sequence=PALETTE))
    else: st.info("At least two numeric features are needed.")
with tabs[4]:
    st.subheader("Trends over time")
    if dates:
        d=st.selectbox("Date feature",dates)
        freq=st.selectbox("Granularity",["D","W","MS","QS"],format_func=lambda x:dict(D="Daily",W="Weekly",MS="Monthly",QS="Quarterly")[x])
        measure=st.selectbox("Measure",["Row count"]+numeric)
        temp=df.dropna(subset=[d]).set_index(d)
        if measure=="Row count": trend=temp.resample(freq).size().rename("Value")
        else:
            agg=st.selectbox("Aggregation",["mean","sum","median","min","max"])
            trend=clean_display(temp[measure]).resample(freq).agg(agg)
        chart(px.area(pd.DataFrame({"Period":trend.index,"Value":trend.to_numpy()}),x="Period",y="Value",title=f"{measure} over time",color_discrete_sequence=PALETTE))
    else: st.info("Select a date column in the sidebar to unlock time-series analysis.")
with tabs[5]:
    st.subheader("Build your own visualization")
    kind=st.selectbox("Chart type",["Scatter","Bar (aggregated)","Box","Histogram","Donut"])
    if kind=="Scatter" and numeric:
        x=st.selectbox("X",numeric,key="sx"); y=st.selectbox("Y",numeric,key="sy")
        chart(px.scatter(df.sample(min(10000,len(df)),random_state=42),x=x,y=y,color_discrete_sequence=PALETTE))
    elif kind=="Histogram" and numeric:
        x=st.selectbox("Feature",numeric,key="hist")
        chart(px.histogram(clean_display(df[x]).dropna().to_frame(),x=x,color_discrete_sequence=PALETTE))
    elif kind in ["Bar (aggregated)","Box","Donut"] and cats:
        group=st.selectbox("Group",cats,key="group")
        if kind=="Donut":
            counts=df[group].astype("string").fillna("(Missing)").value_counts().head(15).rename_axis("Category").reset_index(name="Count")
            chart(px.pie(counts,names="Category",values="Count",hole=.65,title="Top 15 categories",color_discrete_sequence=PALETTE))
        elif numeric:
            value=st.selectbox("Value",numeric,key="value")
            top=df[group].value_counts().head(20).index
            subset=df[df[group].isin(top)]
            if kind=="Box": chart(px.box(subset.sample(min(len(subset),10000),random_state=42),x=group,y=value,color_discrete_sequence=PALETTE))
            else:
                agg=st.selectbox("Aggregation",["sum","mean","median","count"])
                table=subset.groupby(group,observed=True)[value].agg(agg).reset_index()
                chart(px.bar(table,x=group,y=value,color_discrete_sequence=PALETTE))
        else: st.info("Choose a chart that does not require numeric values.")
    else: st.info("This chart requires compatible feature types.")
with tabs[6]:
    st.subheader("Explore and export")
    st.dataframe(df.head(5000),width="stretch")
    st.caption("Preview limited to 5,000 rows. Download includes all filtered rows.")
    st.download_button("Download filtered CSV",df.to_csv(index=False),"filtered_data.csv","text/csv")
st.caption("DataLens • Local analysis • Interactive Plotly charts")
