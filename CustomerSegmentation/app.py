import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from PIL import Image

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# ======================================================
# SIMPLE LOGIN & LOGOUT (DROP-IN BLOCK)
# ======================================================

# Demo users (change later if needed)
USER_CREDENTIALS = {
    "user": "user123",
}

# Session state setup
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None


def login_logout():
    # --------- LOGIN SCREEN ----------
    if not st.session_state.logged_in:
        st.title("🔐 Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if (
                    username in USER_CREDENTIALS
                    and USER_CREDENTIALS[username] == password
                ):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        st.stop()

    # --------- LOGOUT (SIDEBAR) ----------
    st.sidebar.success(f"👤 Logged in as {st.session_state.username}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

login_logout()

banner_img = Image.open("marketing_banner.png")


# SIDEBAR NAVIGATION
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["About", "RFM Analysis", "Customer Segmentation",  "Personalized Recommendations", "Purchase Intent Prediction"]
)

@st.cache_data
def load_data():
    return pd.read_csv("marketing_campaign_c.csv")

df = load_data()

# ===============================
# RFM FEATURE ENGINEERING FUNCTION
# ===============================
@st.cache_data
def create_rfm(df):
    rfm = df.copy()
    rfm["Frequency"] = (
        rfm["NumWebPurchases"]
        + rfm["NumCatalogPurchases"]
        + rfm["NumStorePurchases"]
    )
    rfm["Monetary"] = (
        rfm["MntWines"]
        + rfm["MntFruits"]
        + rfm["MntMeatProducts"]
        + rfm["MntFishProducts"]
        + rfm["MntSweetProducts"]
        + rfm["MntGoldProds"]
    )
    return rfm[["ID", "Recency", "Frequency", "Monetary"]]

# ===============================
# RFM FEATURE ENGINEERING FUNCTION
# ===============================
@st.cache_data
def create_rfm(df):
    rfm = df.copy()
    rfm["Frequency"] = (
        rfm["NumWebPurchases"]
        + rfm["NumCatalogPurchases"]
        + rfm["NumStorePurchases"]
    )
    rfm["Monetary"] = (
        rfm["MntWines"]
        + rfm["MntFruits"]
        + rfm["MntMeatProducts"]
        + rfm["MntFishProducts"]
        + rfm["MntSweetProducts"]
        + rfm["MntGoldProds"]
    )
    return rfm[["ID", "Recency", "Frequency", "Monetary"]]

st.sidebar.subheader("⬇️ Download Dataset")

with open("marketing_campaign_c.csv", "rb") as file:
    st.sidebar.download_button(
        label="Download Dataset (CSV)",
        data=file,
        file_name="marketing_campaign_c.csv",
        mime="text/csv"
    )
st.markdown("""
            
<style>
/* Page background */
.stApp {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #312e81, #4f46e5);
    color: white;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: orange;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

/* Success / info boxes */
.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ABOUT PAGE
if page == "About":

    st.markdown("""
    <style>
    .dashboard-card {
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        padding: 50px;
        border-radius: 24px;
        box-shadow: 0px 12px 30px rgba(0,0,0,0.10);
        margin-top: 10px;
    }
                
    /* Remove Streamlit top header */
    header[data-testid="stHeader"] {
    display: none;
    }

    /* Remove extra top padding */
    .block-container {
     padding-top: 1rem;
    }
     </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
        <h1 style="font-size:38px; font-weight:800; color:#1e3a8a;">
            📈 Customer Segmentation & Recommendation System
        </h1>
                
        <p  style="font-size:14px; color:#475569; line-height:1.7;">
            Modern e-commerce businesses generate large volumes of customer data,
            but many struggle to transform this data into actionable insights.
            This system addresses that challenge by converting raw transactional
            data into meaningful customer segments, behavioural predictions,
            and personalized marketing recommendations.
        </p>

        <p style="font-size:14px; color:#475569; line-height:1.7;">
           By leveraging RFM analysis, machine learning models, and interactive
           visual analytics, this dashboard enables marketing teams and business
           decision-makers to improve customer retention, increase campaign
           effectiveness, and optimize marketing resource allocation.
        </p>

                    
        <p style="font-size:14px; color:#475569;">
            Made by chong zhe 0135063 who is currently taking fyp 2.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.image(banner_img, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.title("Introduction")

    st.markdown("""
    This dashboard applies **Exploratory Data Analysis (EDA)**,  
    **RFM analysis**, and **machine learning techniques**
    to analyze customer behaviour and support **data-driven marketing strategies**.
    """)

    # =====================================================
    # DATA PREPARATION
    # =====================================================
    df_eda = df.copy()
    df_eda["Age"] = 2014 - df_eda["Year_Birth"]
    df_eda = df_eda[(df_eda["Age"] >= 18) & (df_eda["Age"] <= 90)]

    # =====================================================
    # SUMMARY CARDS
    # =====================================================
    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", df_eda.shape[0])
    col2.metric("Features", df_eda.shape[1])
    col3.metric("Campaign Response (%)", f"{df_eda['Response'].mean()*100:.1f}%")
    col4.metric("Avg Recency (Days)", int(df_eda["Recency"].mean()))

    st.divider()

    # =====================================================
    # TABBED EDA LAYOUT
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "👥 Demographics",
        "📣 Campaign Performance",
        "🛒 Purchase Behavior"
    ])

    # =====================================================
    # DEMOGRAPHICS TAB
    # =====================================================
    with tab1:
        st.subheader("Customer Demographics")

        col1, col2 = st.columns(2)

        # AGE GROUPS
        age_bins = pd.cut(
            df_eda["Age"],
            bins=[18, 30, 40, 50, 60, 70, 90],
            labels=["18–30", "31–40", "41–50", "51–60", "61–70", "70+"]
        )

        age_counts = age_bins.value_counts().sort_index().reset_index()
        age_counts.columns = ["Age Group", "Customers"]

        fig_age = px.bar(
            age_counts,
            x="Age Group",
            y="Customers",
            title="Customer Distribution by Age Group",
            color="Customers",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig_age, use_container_width=True)
        
        st.markdown("""
        **Age Group Distribution**

        Age is a key factor influencing purchasing preferences, spending capacity,
        and responsiveness to marketing campaigns. Analyzing age groups allows
        businesses to:

        - Identify the most active customer age segments
        - Design age-appropriate promotions and product recommendations
        - Detect underrepresented age groups with growth potential
        """)


        # EDUCATION
        edu_counts = df_eda["Education"].value_counts().reset_index()
        edu_counts.columns = ["Education Level", "Customers"]

        fig_edu = px.bar(
            edu_counts,
            x="Education Level",
            y="Customers",
            title="Customer Distribution by Education Level",
            color="Customers",
            color_continuous_scale="Greens"
        )

        st.plotly_chart(fig_edu, use_container_width=True)

        col3, col4 = st.columns(2)
        st.markdown("""
        **Education Level Distribution**

        Education level often correlates with income level, purchasing behaviour,
        and responsiveness to marketing campaigns. This analysis helps businesses:

        - Understand how education impacts customer composition
        - Identify customer segments likely to respond to premium or value-based offers
        - Design marketing content that matches customer expectations and preferences
        """)

        # MARITAL STATUS
        marital_counts = df_eda["Marital_Status"].value_counts().reset_index()
        marital_counts.columns = ["Marital Status", "Customers"]

        fig_marital = px.bar(
            marital_counts,
            x="Marital Status",
            y="Customers",
            title="Customer Distribution by Marital Status",
            color="Customers",
            color_continuous_scale="Purples"
        )

        st.plotly_chart(fig_marital, use_container_width=True)
        st.markdown("""
        **Marital Status Distribution**

        Marital status provides insights into customer household structure
        and lifestyle, which often influence purchasing behaviour and
        product preferences.

        Analyzing marital status helps businesses to:
        - Understand differences in spending patterns between single and family households
        - Design targeted promotions for different life stages
        - Identify customer groups likely to respond to family-oriented or individual-focused products
        - Support more precise customer segmentation and personalization
        """)


    # =====================================================
    # CAMPAIGN TAB
    # =====================================================
    with tab2:
        st.subheader("Marketing Campaign Performance")

    # ACCEPTANCE RATE BY CAMPAIGN
        campaign_cols = [
             "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
             "AcceptedCmp4", "AcceptedCmp5"
        ]

        campaign_rates = df_eda[campaign_cols].mean().reset_index()
        campaign_rates.columns = ["Campaign", "Acceptance Rate"]

        fig_campaign = px.bar(
            campaign_rates,
            x="Campaign",
            y="Acceptance Rate",
            title="Campaign Acceptance Rates",
            color="Acceptance Rate",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(fig_campaign, use_container_width=True)

        col1, col2 = st.columns(2)
        st.markdown("""
        **Acceptance Rate of Previous Marketing Campaigns**

        This analysis compares customer acceptance rates across five
        previous marketing campaigns.
        
        - **AcceptedCmp1 to AcceptedCmp5** indicate whether a customer accepted
        one of five previous marketing campaigns.
        - **Response** represents whether the customer accepted the
        most recent marketing campaign.

        Each campaign was conducted at a different time and may vary in
        promotion type, communication strategy, or product focus.

        Differences in acceptance rates indicate varying levels of customer
        engagement over time. Higher acceptance suggests stronger customer
        interest or better alignment between the campaign and customer needs.

        These insights help businesses evaluate past marketing performance
        and identify patterns in customer responsiveness.
        """)

    # RESPONSE BY EDUCATION
        edu_response = df_eda.groupby("Education")["Response"].mean().reset_index()

        fig_edu_resp = px.bar(
            edu_response,
            x="Education",
            y="Response",
            title="Response Rate by Education Level",
            color="Response",
            color_continuous_scale="Teal"
        )

        st.plotly_chart(fig_edu_resp, use_container_width=True)
        st.markdown("""
       **Campaign Response by Education Level**

       This analysis examines how customer responses to the most recent
       marketing campaign vary across different education levels.

       Education level serves as a demographic indicator that may be
       associated with differences in purchasing behaviour, awareness,
       or engagement with marketing content.

       The purpose of this analysis is to identify whether certain
       education groups show higher responsiveness, supporting more
       targeted and informed marketing strategies.
       """)


    # RESPONSE BY INCOME
        fig_income_resp = px.box(
            df_eda,
            x="Response",
            y="Income",
            title="Income Distribution by Campaign Response"
        )

        st.plotly_chart(fig_income_resp, use_container_width=True)

        st.markdown("""
       **Income Distribution by Campaign Response**

       This analysis compares income distributions between customers who
       responded to the most recent campaign and those who did not.

       Income is analyzed as a contextual factor that may influence
       purchasing power and campaign engagement. The goal is not to infer
       causation, but to observe whether noticeable income differences
       exist between responder and non-responder groups.

       Such insights can support more informed segmentation and campaign targeting.
       """)

    # =====================================================
    # PURCHASE BEHAVIOR TAB
    # =====================================================
    with tab3:
        st.subheader("Customer Purchase Behavior")

    # PRODUCT SPENDING
        spending_cols = [
            "MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"
        ]

        spending_sum = df_eda[spending_cols].sum().reset_index()
        spending_sum.columns = ["Product Category", "Total Spending"]

        fig_spending = px.bar(
            spending_sum,
            x="Product Category",
            y="Total Spending",
            title="Total Spending by Product Category",
            color="Total Spending",
            color_continuous_scale="Oranges"
        )

        st.plotly_chart(fig_spending, use_container_width=True)

        col1, col2 = st.columns(2)
        st.markdown("""
        **Total Spending by Product Category**

        Analyzing spending across product categories reveals which products
        generate the most revenue. This insight supports:

        - Strategic product prioritization
        - Marketing focus on high-revenue categories
        - Identification of cross-selling opportunities
        """)


    # PURCHASE CHANNELS
        channel_data = df_eda[[
             "NumWebPurchases",
             "NumCatalogPurchases",
             "NumStorePurchases"
        ]].sum().reset_index()

        channel_data.columns = ["Channel", "Total Purchases"]

        fig_channels = px.bar(
            channel_data,
            x="Channel",
            y="Total Purchases",
            title="Purchases by Channel",
            color="Total Purchases",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig_channels, use_container_width=True)
        st.markdown("""
        **Purchases by Channel**

        This analysis compares customer purchasing behaviour across
        web, catalog, and physical store channels. It helps businesses:

        - Understand preferred shopping channels
        - Identify opportunities to strengthen underperforming channels
        - Align omnichannel marketing strategies
        """)


    # WEB VISITS VS PURCHASES
        fig_web = px.scatter(
            df_eda,
            x="NumWebVisitsMonth",
            y="NumWebPurchases",
            title="Web Visits vs Web Purchases",
            opacity=0.5
        )

        st.plotly_chart(fig_web, use_container_width=True)

        st.markdown("""
        **Web Visits vs Web Purchases**

        This analysis examines whether frequent website visits translate
        into actual purchases. It helps businesses:

        - Identify conversion gaps in the online shopping experience
        - Detect potential issues such as pricing, usability, or product relevance
        - Improve digital marketing and website optimization strategies
        """)

# RFM ANALYSIS PAGE
elif page == "RFM Analysis":
    st.title("📊 RFM Analysis")

    st.markdown("""
    ### 🔍 What is RFM Analysis?
    **RFM (Recency, Frequency, Monetary)** is a widely used customer analysis method
    that evaluates customer value based on purchasing behaviour:

    - **Recency (R):** is taken directly from the dataset as *days since last purchase
    - **Frequency (F):** aggregates customer purchase counts across all sales channels.
    - **Monetary (M):** aggregates total spending across all product categories.

    This ensures the RFM values accurately represent customer purchasing behaviour.
                
    Customers who purchase **recently**, **frequently**, and **spend more**
    are generally more valuable to the business.
    """)

    # RFM FEATURE ENGINEERING
    rfm = df.copy()

    rfm["Frequency"] = (
        rfm["NumWebPurchases"]
        + rfm["NumCatalogPurchases"]
        + rfm["NumStorePurchases"]
    )

    rfm["Monetary"] = (
        rfm["MntWines"]
        + rfm["MntFruits"]
        + rfm["MntMeatProducts"]
        + rfm["MntFishProducts"]
        + rfm["MntSweetProducts"]
        + rfm["MntGoldProds"]
    )

    rfm = rfm[["ID", "Recency", "Frequency", "Monetary"]]
    st.subheader("🔍 Sample RFM Records (Verification)")
    st.dataframe(rfm.head(10), hide_index=True)

    st.markdown("""
    ### 🧮 RFM Calculation
    - **Recency** is provided directly in the dataset (days since last purchase)
    - **Frequency** is calculated as the total number of purchases across channels
    - **Monetary** represents total customer spending across product categories
    """)

    # SUMMARY METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Recency (days)", int(rfm["Recency"].mean()))
    col2.metric("Average Frequency", int(rfm["Frequency"].mean()))
    col3.metric("Average Monetary Value", f"{rfm['Monetary'].mean():.2f}")

    # DISTRIBUTIONS
    # ===============================
    st.subheader("📈 RFM Distributions")

    tab_r, tab_f, tab_m = st.tabs([
        "⏱ Recency",
        "🔁 Frequency",
        "💰 Monetary"
    ])

    with tab_r:
         fig_r = px.histogram(
         rfm,
         x="Recency",
         nbins=30,
         title="Recency Distribution",
         color_discrete_sequence=["#3B82F6"]
        )
         st.plotly_chart(fig_r, use_container_width=True)
         st.caption("Lower recency indicates more recent customer activity.")

    with tab_f:
         fig_f = px.histogram(
         rfm,
         x="Frequency",
         nbins=30,
         title="Frequency Distribution",
         color_discrete_sequence=["#10B981"]
        )
         st.plotly_chart(fig_f, use_container_width=True)
         st.caption("Higher frequency reflects more frequent purchases.")

    with tab_m:
         fig_m = px.histogram(
         rfm,
         x="Monetary",
         nbins=30,
         title="Monetary Distribution",
         color_discrete_sequence=["#F59E0B"]
         )
         st.plotly_chart(fig_m, use_container_width=True)
         st.caption("Higher monetary value represents higher customer spending.")




# CUSTOMER SEGMENTATION PAGE
elif page == "Customer Segmentation":
    st.title("🧠 Customer Segmentation")

    st.markdown("""
    Customer segmentation groups customers with similar purchasing behaviour.
    This enables businesses to personalize marketing strategies, retain high-value customers,
    and re-engage customers who are at risk of churn.
    """)

    # =====================================================
    # RFM PREPARATION
    # =====================================================
    rfm = df.copy()

    rfm["Frequency"] = (
        rfm["NumWebPurchases"]
        + rfm["NumCatalogPurchases"]
        + rfm["NumStorePurchases"]
    )

    rfm["Monetary"] = (
        rfm["MntWines"]
        + rfm["MntFruits"]
        + rfm["MntMeatProducts"]
        + rfm["MntFishProducts"]
        + rfm["MntSweetProducts"]
        + rfm["MntGoldProds"]
    )

    rfm = rfm[["ID", "Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Clustering",
        "📊 Cluster Profiling",
        "🕸 Radar Comparison",
        "🧾 Business Interpretation"
    ])

    # =====================================================
    # TAB 1: CLUSTERING
    # =====================================================
    with tab1:
        st.subheader("Clustering Configuration")
        st.markdown("""
    ### ⚙️ Clustering Algorithm Selection

    Three clustering algorithms are included in this system to allow
    comparative analysis of customer segmentation results, as different
    algorithms capture different data characteristics.

    - **Gaussian Mixture Model (GMM)**  
    GMM is a probabilistic clustering approach that assumes customers
    are generated from a mixture of Gaussian distributions. It is suitable
    for customer segmentation because it allows soft cluster boundaries
    and can model overlapping customer behaviours.

    - **Hierarchical Clustering**  
    Hierarchical clustering builds clusters based on similarity structure
    without assuming a predefined distribution. It provides interpretable
    segmentation and helps understand customer relationships at different
    granularity levels.

    - **DBSCAN**  
    DBSCAN is a density-based clustering algorithm that can identify
    irregularly shaped clusters and detect noise or outlier customers.
    This is useful for identifying atypical customer behaviours that
    do not belong to any major segment.

    Using multiple clustering approaches enables robust evaluation of
    segmentation quality and supports more informed model selection.
    """)


        model_choice = st.selectbox(
            "Choose clustering algorithm",
            ["Gaussian Mixture Model", "Hierarchical Clustering", "DBSCAN"]
        )

        if model_choice in ["Gaussian Mixture Model", "Hierarchical Clustering"]:
            k = st.slider("Number of clusters", 2, 6, 3)

        if model_choice == "Gaussian Mixture Model":
            model = GaussianMixture(n_components=k, random_state=42)
            rfm["Cluster"] = model.fit_predict(rfm_scaled)
            sil_score = silhouette_score(rfm_scaled, rfm["Cluster"])

        elif model_choice == "Hierarchical Clustering":
            model = AgglomerativeClustering(n_clusters=k)
            rfm["Cluster"] = model.fit_predict(rfm_scaled)
            sil_score = silhouette_score(rfm_scaled, rfm["Cluster"])

        else:
            eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5)
            min_samples = st.slider("min_samples", 3, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            rfm["Cluster"] = model.fit_predict(rfm_scaled)

            if len(set(rfm["Cluster"])) > 1:
                sil_score = silhouette_score(rfm_scaled, rfm["Cluster"])
            else:
                sil_score = None

        if sil_score is not None:
            st.success(f"Silhouette Score: {sil_score:.3f}")
        else:
            st.warning("Silhouette Score not available for this configuration.")

        st.subheader("Cluster Size Distribution")

        cluster_counts = rfm["Cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster", "Number of Customers"]

        # Convert Cluster to string (categorical)
        cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)

        fig_cluster_size = px.bar(
            cluster_counts,
            x="Cluster",
            y="Number of Customers",
            title="Customers per Cluster",
            color="Number of Customers",
            color_continuous_scale="Blues"
        )

        fig_cluster_size.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Number of Customers"
        )
        fig_cluster_size.update_xaxes(type="category")
        st.plotly_chart(fig_cluster_size, use_container_width=True)

        st.caption("""
        This chart shows the distribution of customers across the identified
        clusters. Balanced cluster sizes indicate stable segmentation, while
        extremely small clusters may represent niche or atypical customer groups.
        """)

    # =====================================================
    # TAB 2: CLUSTER PROFILING
    # =====================================================
    with tab2:
        st.subheader("Average RFM Values per Cluster")

        profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        st.dataframe(profile)

        st.markdown("""
        **Average RFM Values per Cluster**

        This table presents the mean Recency, Frequency, and Monetary values
        for each customer cluster. It provides a direct view of how customer
        behaviour differs across segments based on actual purchasing data.
        """)


        st.subheader("Normalized Cluster Profile (0–1 Scale)")

        profile_norm = profile.copy()

        # Invert Recency (lower is better)
        profile_norm["Recency"] = (
            profile["Recency"].max() - profile["Recency"]
        ) / (profile["Recency"].max() - profile["Recency"].min())

        profile_norm["Frequency"] = (
            profile["Frequency"] - profile["Frequency"].min()
        ) / (profile["Frequency"].max() - profile["Frequency"].min())

        profile_norm["Monetary"] = (
            profile["Monetary"] - profile["Monetary"].min()
        ) / (profile["Monetary"].max() - profile["Monetary"].min())

        st.dataframe(profile_norm.round(2))
        st.markdown("""
        **Normalized Cluster Profile (0–1 Scale)**

        RFM values are normalized to a 0–1 scale to enable fair comparison
        across clusters. Recency is inverted so that higher values represent
        more recent customer activity, making interpretation consistent
        across all dimensions.
        """)

        st.info(
            "RFM values are normalized to enable fair comparison across clusters. "
            "Recency is inverted so higher values indicate more recent activity."
        )

    # =====================================================
    # TAB 3: RADAR CHART
    # =====================================================
    with tab3:
        st.subheader("Cluster Comparison Using Radar Chart (click right side cluster 0, 1, 2 to manipulate)")

        categories = ["Recency", "Frequency", "Monetary"]
        fig_radar = go.Figure()

        for cluster in profile_norm.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=profile_norm.loc[cluster, categories],
                theta=categories,
                fill="toself",
                name=f"Cluster {cluster}"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Radar Chart of Customer Clusters (RFM)"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        st.caption(
            "The radar chart highlights behavioural differences across clusters, "
            "making it easier to interpret and label customer segments."
        )

    # =====================================================
    # TAB 4: BUSINESS INTERPRETATION
    # =====================================================
    with tab4:
        st.subheader("Cluster Interpretation & Marketing Strategy")

        segment_summary = []

        for cluster in profile.index:
            rec = profile.loc[cluster, "Recency"]
            freq = profile.loc[cluster, "Frequency"]
            mon = profile.loc[cluster, "Monetary"]

            if rec < profile["Recency"].median() and freq > profile["Frequency"].median():
                label = "High-Value / Loyal Customers"
                strategy = "Loyalty rewards, exclusive offers, premium products"

            elif rec > profile["Recency"].median():
                label = "At-Risk Customers"
                strategy = "Re-engagement campaigns, discounts, reminder emails"

            else:
                label = "Potential Customers"
                strategy = "Personalized recommendations and upselling strategies"

            segment_summary.append([cluster, label, strategy])

        summary_df = pd.DataFrame(
            segment_summary,
            columns=["Cluster", "Segment Label", "Recommended Marketing Strategy"]
        )

        st.dataframe(summary_df, hide_index=True)

        st.success(
            "These interpretations translate clustering results into actionable business insights, "
            "supporting data-driven marketing decisions."
        )

elif page == "Personalized Recommendations":
    st.title("🎯 Personalized Recommendation System")

    st.markdown("""
    This module generates **actionable marketing recommendations** by combining:
    - Customer segmentation results
    - RFM behavioural analysis
    - Purchase intent prediction

    The goal is to help decision-makers design **targeted, data-driven marketing strategies**.
    """)

    # PREPARE DATA (RFM + SEGMENT)
    rec_df = df.copy()

    rec_df["Frequency"] = (
        rec_df["NumWebPurchases"]
        + rec_df["NumCatalogPurchases"]
        + rec_df["NumStorePurchases"]
    )

    rec_df["Monetary"] = (
        rec_df["MntWines"]
        + rec_df["MntFruits"]
        + rec_df["MntMeatProducts"]
        + rec_df["MntFishProducts"]
        + rec_df["MntSweetProducts"]
        + rec_df["MntGoldProds"]
    )

    rec_df = rec_df[["ID", "Recency", "Frequency", "Monetary", "Response"]]


    def segment_customer(row):
        if row["Recency"] <= 30 and row["Frequency"] > 10:
            return "Loyal Customers"
        elif row["Recency"] > 90:
            return "At-Risk Customers"
        else:
            return "Potential Customers"

    rec_df["Segment"] = rec_df.apply(segment_customer, axis=1)


    X = rec_df[["Recency", "Frequency", "Monetary"]]
    y = rec_df["Response"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    rec_df["Purchase_Intent_Prob"] = model.predict_proba(X_scaled)[:, 1]

    def generate_recommendation(row):
        if row["Segment"] == "Loyal Customers":
            if row["Purchase_Intent_Prob"] > 0.7:
                return "Offer premium products, loyalty rewards, and early-access promotions"
            else:
                return "Maintain engagement with personalized appreciation messages"

        elif row["Segment"] == "Potential Customers":
            if row["Purchase_Intent_Prob"] > 0.5:
                return "Recommend popular products and limited-time offers"
            else:
                return "Provide onboarding promotions and product discovery campaigns"

        else:  # At-Risk Customers
            if row["Purchase_Intent_Prob"] < 0.3:
                return "Send strong re-engagement discounts or win-back campaigns"
            else:
                return "Send reminder emails and personalized incentives"

    rec_df["Recommendation"] = rec_df.apply(generate_recommendation, axis=1)

    st.subheader("🔍 Explore Recommendations")

    selected_segment = st.selectbox(
        "Filter by Customer Segment",
        ["All", "Loyal Customers", "Potential Customers", "At-Risk Customers"]
    )

    if selected_segment != "All":
        display_df = rec_df[rec_df["Segment"] == selected_segment]
    else:
        display_df = rec_df

    st.dataframe(
        display_df[[
            "ID",
            "Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "Purchase_Intent_Prob",
            "Recommendation"
        ]].sort_values(by="Purchase_Intent_Prob", ascending=False),
         hide_index=True,
         use_container_width=True
    )

    st.subheader("📊 Recommendation Insights")

    summary = display_df.groupby("Segment").agg({
        "ID": "count",
        "Purchase_Intent_Prob": "mean"
    }).rename(columns={"ID": "Number of Customers"})

    st.dataframe(summary)

    st.info("""
    These recommendations enable marketers to:
    - Prioritize high-value customers
    - Reduce churn among at-risk customers
    - Allocate marketing budgets more efficiently
    """)

    st.subheader("📈 Average Purchase Intent by Segment")

    avg_prob = rec_df.groupby("Segment")["Purchase_Intent_Prob"].mean().reset_index()

    fig_avg = px.bar(
        avg_prob,
        x="Segment",
        y="Purchase_Intent_Prob",
        title="Average Purchase Intent Probability per Segment",
        color="Segment",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(fig_avg, use_container_width=True)

    st.caption(
       "➡️ Want to go further? "
       "Proceed to the **Purchase Intent Prediction** page to predict whether "
       "individual customers are likely to respond to a marketing campaign."
    )


# PURCHASE INTENT PREDICTION PAGE
elif page == "Purchase Intent Prediction":
    st.title("🛒 Purchase Intent Prediction")
    st.caption(
    "This page moves from model evaluation to practical usage — "
    "ending with the ability to predict an individual customer’s purchase intention."
)
    st.markdown("""
    This module predicts whether a customer is likely to respond to a marketing campaign
    using **RFM behavioural features**. Multiple machine learning models are evaluated
    to identify the most suitable approach.
    """)

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    df_pred = df.copy()

    df_pred["Frequency"] = (
        df_pred["NumWebPurchases"]
        + df_pred["NumCatalogPurchases"]
        + df_pred["NumStorePurchases"]
    )

    df_pred["Monetary"] = (
        df_pred["MntWines"]
        + df_pred["MntFruits"]
        + df_pred["MntMeatProducts"]
        + df_pred["MntFishProducts"]
        + df_pred["MntSweetProducts"]
        + df_pred["MntGoldProds"]
    )

    X = df_pred[["Recency", "Frequency", "Monetary"]]
    y = df_pred["Response"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "⚙️ Model Training & Comparison",
        "📊 Feature Importance",
        "🔮 Individual Prediction"
    ])

    # =====================================================
    # TAB 1: MODEL TRAINING & COMPARISON
    # =====================================================
    with tab1:
        st.subheader("Model Performance Comparison")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            results.append([name, acc, auc])

        results_df = pd.DataFrame(
            results,
            columns=["Model", "Accuracy", "ROC-AUC"]
        ).sort_values(by="ROC-AUC", ascending=False)

        st.dataframe(results_df, hide_index=True)

        st.info(
            "Logistic Regression provides interpretability, while tree-based models "
            "capture non-linear relationships and often achieve higher predictive performance."
        )

    # =====================================================
    # TAB 2: FEATURE IMPORTANCE
    # =====================================================
    with tab2:
        st.subheader("Feature Importance Analysis")

        model_choice = st.selectbox(
            "Select model for feature importance",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )

        model = models[model_choice]
        model.fit(X_train, y_train)

        if model_choice == "Logistic Regression":
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig_imp = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            title=f"Feature Importance – {model_choice}",
            color="Importance",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig_imp, use_container_width=True)

        st.caption(
            "Frequency and Monetary features consistently show strong influence "
            "on campaign response prediction across models."
        )

    # =====================================================
    # TAB 3: INDIVIDUAL PREDICTION
    # =====================================================
    with tab3:
        st.subheader("Predict Customer Purchase Intent")

        st.markdown("""
        ### 🤖 Model Selection Rationale

       Three machine learning models are used in this module to balance
       **interpretability**, **predictive performance**, and **model robustness**.

       - **Logistic Regression**  
       Logistic Regression serves as a baseline model due to its simplicity
       and interpretability. It allows clear understanding of how Recency,
       Frequency, and Monetary features influence campaign response, making
       it suitable for explaining results to business stakeholders.

       - **Random Forest**  
       Random Forest captures non-linear relationships and interactions
       between RFM features that linear models may miss. It is robust to
       noise and often achieves higher predictive accuracy in customer
       behaviour prediction tasks.

       - **Gradient Boosting**  
       Gradient Boosting incrementally improves prediction by correcting
       previous model errors. It is effective at learning complex patterns
       and is included to evaluate whether advanced ensemble methods provide
       superior performance.
       """)

        model_choice = st.selectbox(
            "Select prediction model",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )

        selected_model = models[model_choice]
        selected_model.fit(X_train, y_train)

        r = st.slider("Recency (days)", 0, 200, 30)
        f = st.slider("Frequency", 0, 50, 5)
        m = st.slider("Monetary Value", 0, 5000, 500)

        input_df = pd.DataFrame(
            [[r, f, m]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        input_scaled = scaler.transform(input_df)

        if st.button("Predict Purchase Intent"):
            pred = selected_model.predict(input_scaled)[0]
            prob = selected_model.predict_proba(input_scaled)[0][1]

            if pred == 1:
                st.success(f"✅ High Purchase Intent (Probability: {prob:.2f})")
                st.info("Recommended Action: Send personalized promotions.")
            else:
                st.warning(f"⚠ Low Purchase Intent (Probability: {prob:.2f})")
                st.info("Recommended Action: Use discounts or re-engagement campaigns.")