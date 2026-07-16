import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import joblib
import plotly.graph_objects as go
from streamlit import button

saved_mdl=joblib.load("risk_model.pkl")
model=saved_mdl["model"]
feature_columns=saved_mdl["feature_columns"]

input_data = pd.DataFrame(
    [[0] * len(feature_columns)],
    columns=feature_columns
)


st.set_page_config(page_title="SafeRoadIndiaAnalyticsDashboard",page_icon="🚗",layout="wide")
df=pd.read_csv("Final_accident_dataset.csv")


st.sidebar.title("📊 Visualization Filters")

#year filter
year_options = ["All"] + sorted(df["year"].unique().tolist())
selected_year = st.sidebar.selectbox(
    "Select Year",
    year_options
)

#state filter
state_options = ["All"] + sorted(df["state"].unique().tolist())
selected_state = st.sidebar.selectbox(
    "Select State",
    state_options
)

#city filter
if selected_state == "All":
    city_options = ["All"] + sorted(df["city"].unique().tolist())
else:
    city_options = ["All"] + sorted(
        df[df["state"] == selected_state]["city"].unique().tolist()
    )

selected_city = st.sidebar.selectbox(
    "Select City",
    city_options
)

st.sidebar.divider()


##ML
st.sidebar.title("🔮 Accident Risk Predictor")

temperature=st.sidebar.slider(
    "Temperature (°C)",
    0,
    50,
    25
)
input_data["temperature"] = temperature

hour=st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)
input_data["hour"]=hour

lanes = st.sidebar.selectbox(
    "Number of Lanes",
    [1,2,3,4,5,6,7,8]
)
input_data["lanes"]=lanes

is_peak_hour = st.sidebar.selectbox(
    "Peak Hour",
    [0,1],
    format_func=lambda x:"Yes" if x else "No"
)
input_data["is_peak_hour"]=is_peak_hour

is_weekend = st.sidebar.selectbox(
    "Weekend",
    [0,1],
    format_func=lambda x:"Yes" if x else "No"
)
input_data["is_weekend"] = is_weekend

traffic_signal = st.sidebar.selectbox(
    "Traffic Signal",
    [0,1],
    format_func=lambda x:"Present" if x else "Absent"
)
input_data["traffic_signal"] = traffic_signal

festival = st.sidebar.selectbox(
    "Festival",
    [0,1],
    format_func=lambda x:"Yes" if x else "No"
)
input_data["festival"] =festival


weather = st.sidebar.selectbox(
    "Weather",
    [
        "Clear",
        "Fog",
        "Rain"
    ]
)

if weather=="Fog":
    input_data["weather_fog"] = 1
elif weather=="Rain":
    input_data["weather_rain"] = 1

visibility = st.sidebar.selectbox(
    "Visibility",
    [
        "High",
        "Medium",
        "Low"
    ]
)
if visibility=="Low":
    input_data["visibility_low"] = 1
elif visibility=="Medium":
    input_data["visibility_medium"] = 1

traffic_density = st.sidebar.selectbox(
    "Traffic Density",
    [
        "High",
        "Medium",
        "Low"
    ]
)
if traffic_density=="Low":
    input_data["traffic_density_low"] = 1
elif traffic_density=="Medium":
    input_data["traffic_density_medium"] = 1

road_type = st.sidebar.selectbox(
    "Road Type",
    [
        "Highway",
        "Urban",
        "Rural"
    ]
)
if road_type=="Rural":
    input_data["road_type_rural"] = 1
elif road_type=="Urban":
    input_data["road_type_urban"] = 1



#display filtered data
filtered_df = df.copy()

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["year"] == selected_year
    ]

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["state"] == selected_state
    ]

if selected_city != "All":
    filtered_df = filtered_df[
        filtered_df["city"] == selected_city
    ]

st.title("SafeRoadIndia")

st.subheader("Interactive Data Analytics Dashboard, based on Indian Road Accidents(2022–2025) Dataset (Kaggle)")

st.write("Developed by Shriyans Mohanty")



#KPIs
col1,col2,col3,col4=st.columns(4)
with col1:
    st.metric("🚗 Total Accidents",len(filtered_df))
with col2:
    st.metric("⚠ Fatal Accidents",len(filtered_df[filtered_df["accident_severity"]=="fatal"]))
with col3:
    st.metric("📍 States",len(filtered_df["state"].unique()))
with col4:
    st.metric("🏙 Cities",len(filtered_df["city"].unique()))


#st.subheader("Accident datasets")

#display the filtered df
#st.dataframe(filtered_df.head())

#st.write("Rows=",df.shape[0])
#st.write("Columns=",df.shape[1])

#plots
# hotspot map india
st.subheader("Accident Hotspot Density Across India")
st.caption(
    "Darker regions indicate a higher concentration of reported accidents."
)
with open("hotspot_map.html", "r", encoding="utf-8") as f:
    html = f.read()
components.html(html, height=800)
st.divider()


#top5 accident states
df_state_top5=filtered_df["state"].value_counts().head(5)
most_accident_state=filtered_df["state"].value_counts().idxmax()
most_accident_state_no=filtered_df["state"].value_counts().max()

colors2 = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
fig, ax = plt.subplots(figsize=(8,4.1))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
bars = ax.barh(
    df_state_top5.index,
    df_state_top5.values,
    color=colors2
)
ax.bar_label(bars, fontsize=10)
ax.set_title("Top 5 Accident Prone States",fontsize=16,color="#0B3C5D")
ax.set_ylabel("State",fontsize=12)
ax.set_xlabel("Accident Count(2022-2025)",fontsize=12)




#top5 cities
df_city_top5=filtered_df["city"].value_counts().head(5)
most_accident_city=filtered_df["city"].value_counts().idxmax()
most_accident_city_no=filtered_df["city"].value_counts().max()

#colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#76B7B2']
fig2, ax2 = plt.subplots(figsize=(8,4))
fig2.patch.set_facecolor("#FAFAFA")
ax2.set_facecolor("#FAFAFA")
bars2=ax2.barh(df_city_top5.index, df_city_top5.values,color=colors2)
ax2.set_title("Top 5 Accident Prone Cities",fontsize=16,color="#0B3C5D")
ax2.set_xlim(
    0,
    df_city_top5.max() + 300
)
ax2.set_ylabel("City",fontsize=12)
ax2.set_xlabel("Accident Count(2022-2025)",fontsize=12)
ax2.bar_label(bars2)


#weather_sev
weather_sev_df=pd.crosstab(filtered_df["weather"],filtered_df["accident_severity"])
weather_most_accident=filtered_df["weather"].value_counts().idxmax()
weather_most_accident_no=filtered_df["weather"].value_counts().max()

colors5 = ['#B23A48', '#D4A017', '#2D6A4F']

fig3, ax3 = plt.subplots(figsize=(8,4.8))
fig3.patch.set_facecolor("#FAFAFA")
ax3.set_facecolor("#FAFAFA")

weather_sev_df.plot(kind="barh",stacked=False, color=colors5,ax=ax3)

for container in ax3.containers:
    ax3.bar_label(container, fmt='%d')
ax3.set_title("Weather vs Accident Severity",fontsize=16,color="#0B3C5D")
ax3.set_ylabel("Weather type",fontsize=12)
ax3.set_xlabel("Number of Accidents",fontsize=12)
#plt.xticks(rotation=45)
ax3.legend(bbox_to_anchor=(1.02,1),loc="upper left")


#road_sev
road_sev_df=pd.crosstab(filtered_df["road_type"],filtered_df["accident_severity"])
road_most_accident=filtered_df["road_type"].value_counts().idxmax()
road_most_accident_no=filtered_df["road_type"].value_counts().max()

fig4,ax4=plt.subplots(figsize=(8,5))
fig4.patch.set_facecolor("#FAFAFA")
ax4.set_facecolor("#FAFAFA")

road_sev_df.plot(kind="barh",stacked=False, color=colors5,ax=ax4)

for container in ax4.containers:
    ax4.bar_label(container, fmt='%d')
ax4.set_title("Road Type vs Accident Severity",fontsize=16,color="#0B3C5D")
ax4.set_ylabel("Road Type",fontsize=12)
ax4.set_xlabel("Number of Accidents",fontsize=12)
ax4.legend(bbox_to_anchor=(1.02,1),loc="upper left")
#plt.xticks(rotation=45)ax4.legend(bbox_to_anchor=(1.02,1),loc="upper left")


#cause_sev
cause_sev_df=pd.crosstab(filtered_df["cause"],filtered_df["accident_severity"])
fig5,ax5=plt.subplots(figsize=(8,5))
fig5.patch.set_facecolor("#FAFAFA")
ax5.set_facecolor("#FAFAFA")

cause_most_accident=filtered_df["cause"].value_counts().idxmax()
cause_most_accident_no=filtered_df["cause"].value_counts().max()

colors5 = ['#B23A48', '#D4A017', '#2D6A4F']
cause_sev_df.plot(kind="barh",stacked=False, color=colors5,ax=ax5)

for container in ax5.containers:
    ax5.bar_label(container, fmt='%d')

ax5.set_title("Accident Cause vs Severity",fontsize=16,color="#0B3C5D")
ax5.set_xlabel("Number of Accidents",fontsize=12)
ax5.set_ylabel("Accident Cause",fontsize=12)
#plt.xticks(rotation=45)
ax5.legend(bbox_to_anchor=(1.02,1),loc="upper left")


#Year_trend
accident_year = filtered_df["year"].value_counts().sort_index()
year_most_accident=accident_year.idxmax()
year_most_accident_no=accident_year.max()

fig6,ax6=plt.subplots(figsize=(8,3.8))
fig6.patch.set_facecolor("#FAFAFA")
ax6.set_facecolor("#FAFAFA")

ax6.plot(accident_year.index,accident_year.values,label="accidents",color='#1F4E79',marker="o")
ax6.set_title("Year wise accident trend",fontsize=16,color="#0B3C5D")
ax6.set_xlabel("Year",fontsize=12)
ax6.set_ylabel("Accident count",fontsize=12)
ax6.legend()
#plt.xticks(rotation=45)


#Dayxhr_heatmap
day_hour = filtered_df.pivot_table(
    index="day_of_week",
    columns="hour",
    aggfunc="size",
    fill_value=0
)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]
day_hour = day_hour.reindex(day_order)


fig7, ax7 = plt.subplots(figsize=(8,5))
fig7.patch.set_facecolor("#FAFAFA")
ax7.set_facecolor("#FAFAFA")

sns.heatmap(
    day_hour,
    cmap="rocket_r",
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Accident Count"},
    ax=ax7
)

ax7.set_title("Day × Hour Accident Count Heatmap", fontsize=16,color="#0B3C5D")
ax7.set_xlabel("Hour of Day",fontsize=12)
ax7.set_ylabel("Day of Week",fontsize=12)


#pie chart
accident_times=filtered_df["day_time"].value_counts()
most_accident_time=filtered_df["day_time"].value_counts().idxmax()
most_accident_time_no=filtered_df["day_time"].value_counts().max()
fig8,ax8=plt.subplots(figsize=(6,3))
fig8.patch.set_facecolor("#FAFAFA")
ax8.set_facecolor("#FAFAFA")

colors_ = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#76B7B2']
ax8.pie(accident_times,labels=accident_times.index,colors=colors_,autopct="%1.1f%%",startangle=90,wedgeprops={"edgecolor":"white","linewidth":2})
ax8.set_title("Time of accident",fontsize=12,color="#0B3C5D")



#risk gauge
#but=st.sidebar.button("PREDICT",type="primary")
#if but:
risk_pred = model.predict(input_data)[0]
contri=model.coef_*input_data.iloc[0]
contri_df=pd.DataFrame({"coulmn":feature_columns,"contribution":contri})
contri_df=contri_df[contri_df["contribution"]>0.01]
contri_df=contri_df.sort_values(by="contribution",ascending=False)
top3_contri=contri_df.head(3)
#st.sidebar.write(top3_contri)


fig9 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_pred*100,
    number={"suffix": "%"},

    #title={"text": "Predicted Risk Score"},

    gauge={
        "axis": {"range": [0, 100]},

        "bar": {"color": "#0B3C5D"},  # Dark Blue needle/bar

        "steps": [
            {"range": [0, 33], "color": "#2E8B57"},  # Sea Green
            {"range": [33, 66], "color": "#D4A017"},  # Golden
            {"range": [66, 100], "color": "#C0392B"}  # Dark Red
        ]
    }


))


fig9.update_layout(
    height=160,
    margin=dict(
        l=20,
        r=25,
        t=20,
        b=20
    )
)

st.sidebar.plotly_chart(fig9, use_container_width=True)
risk_percent = risk_pred * 100

if risk_percent < 33:
    risk_level = "🟢 LOW RISK"
    color = "#2E8B57"
elif risk_percent < 66:
    risk_level = "🟡 MEDIUM RISK"
    color = "#D4A017"
else:
    risk_level = "🔴 HIGH RISK"
    color = "#C0392B"

st.sidebar.markdown(
    f"""
        <div style='text-align:center; margin-top:-8px;'>
            <span style='font-size:20px; font-weight:700; color:{color};'>
                {risk_level}
            </span>
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )

feature_names = {
        "visibility_low": "🌫 Low Visibility",
        "visibility_medium": "🌤 Medium Visibility",
        "weather_rain": "🌧 Rain",
        "weather_fog": "🌫 Fog",
        "is_peak_hour": "🚗 Peak Hour",
        "festival": "🎉 Festival",
        "traffic_density_low": "🚙 Low Traffic",
        "traffic_density_medium": "🚕 Medium Traffic",
        "traffic_signal": "🚦 Traffic Signal",
        "road_type_urban": "🏙 Urban Road",
        "road_type_rural": "🌾 Rural Road",
        "temperature": "🌡 Temperature",
        "hour": "🕒 Hour",
        "lanes": "🛣 Lanes"
    }

explanations = {
        "visibility_low": "Poor visibility significantly increased the predicted risk.",
        "visibility_medium": "Moderate visibility slightly increased the predicted risk.",
        "weather_rain": "Wet road conditions increased the predicted risk.",
        "weather_fog": "Fog reduced visibility and increased the predicted risk.",
        "is_peak_hour": "Peak-hour traffic increased the predicted risk.",
        "festival": "Festival traffic increased the predicted risk.",
        "traffic_density_low": "Traffic conditions influenced the predicted risk.",
        "traffic_density_medium": "Moderate traffic increased the predicted risk.",
        "road_type_rural": "The rural road environment affected the predicted risk.",
        "road_type_urban": "The urban road environment affected the predicted risk.",
        "traffic_signal": "Traffic signal presence influenced the prediction.",
        "temperature": "Higher temperature contributed to the prediction.",
        "hour": "The selected hour influenced the predicted risk.",
        "lanes": "The number of lanes affected the predicted risk.",
        "is_weekend": "Weekend traffic patterns influenced the prediction."
    }

st.sidebar.markdown("### 🛡 Highest Contributing Factors:")


for _, row in top3_contri.iterrows():
    key = row["coulmn"]  # original name
    display = feature_names.get(key, key)

    st.sidebar.markdown(f"**{display}**")
    st.sidebar.caption(
        explanations.get(key, "Contributed to the prediction.")
    )

if top3_contri.empty:
    st.sidebar.markdown(
            """
            <p style="color:gray; font-size:14px;">
                NO SIGNIFICANT CONTRIBUTING FACTORS FOUND!!.
                <br>
                <i>(Small contribution values ignored)</i>
            </p>
            """,
            unsafe_allow_html=True
    )

#plot col
st.subheader("Geographic Distribution")
st.caption("Identify the states and cities with the highest reported road accident occurrences.")
col1,col2=st.columns(2)

st.subheader("Severity Analysis")
st.caption("Analyze how accident severity varies under different weather conditions and road types.")
col3,col4=st.columns(2)

st.subheader("Cause & Year-Wise Analysis")
st.caption("Understand the leading causes of road accidents along with Severity and examine annual accident trends.")
col5,col6=st.columns(2)

st.subheader("Temporal Analysis")
st.caption("Discover accident patterns across different days, hours, and times of the day.")
col7, col8 = st.columns([2.7,2])
st.info(
    f"💡Key Insight: Most accidents occurred during the {most_accident_time.lower()} time (reported {most_accident_time_no} cases)."
)

with col1:
    st.pyplot(fig)
    st.info(
        f"💡Key Insight: {most_accident_state} recorded the highest number of accidents "
        f"({most_accident_state_no} cases)."
    )
    st.divider()

with col2:
    st.pyplot(fig2)
    st.info(
        f"💡Key Insight: {most_accident_city} emerged as the most accident-prone city ({most_accident_city_no} cases)."
    )
    st.divider()

with col3:
    st.pyplot(fig3)
    st.info(
    f"💡Key Insight: Most reported accidents occurred during {weather_most_accident} weather."
)
    st.divider()

with col4:
    st.pyplot(fig4)
    st.info(
        f"💡Key Insight: {road_most_accident.title()} roads recorded the highest number of accidents ({road_most_accident_no} cases)."
    )
    st.divider()

with col5:
    st.pyplot(fig5)
    st.info(
        f"💡Key Insight: The leading reported cause of accidents was '{cause_most_accident}' ({cause_most_accident_no} cases)."
    )
    st.divider()

with col6:
    st.pyplot(fig6)
    st.info(
        f"💡Key Insight: {year_most_accident} recorded the highest accident count ({year_most_accident_no} incidents)."
    )
    st.divider()
with col7:
    st.pyplot(fig7)

with col8:
    st.pyplot(fig8)

