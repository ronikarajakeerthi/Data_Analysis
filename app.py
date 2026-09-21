

# ============================================================
# ONLINE RETAIL CUSTOMER ANALYSIS
# Python + Streamlit + Plotly + K-Means
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Online Retail Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Retail Customer Analysis")

st.write(
    "Analyze sales, customers, products and customer segments "
    "using Python, Plotly and K-Means Clustering."
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

st.header("📂 Dataset")

try:
    df = pd.read_csv(
        "Online_Retail.csv",
        encoding="latin1"
    )

except FileNotFoundError:

    st.warning(
        "Online_Retail.csv was not found. "
        "Please upload the file."
    )

    uploaded_file = st.file_uploader(
        "Upload Online_Retail.csv",
        type=["csv"]
    )

    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(
        uploaded_file,
        encoding="latin1"
    )


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "CustomerID",
    "Quantity",
    "UnitPrice",
    "Description",
    "InvoiceNo",
    "InvoiceDate",
    "Country"
]

missing_columns = []

for column in required_columns:

    if column not in df.columns:
        missing_columns.append(column)


if len(missing_columns) > 0:

    st.error(
        "The following required columns are missing:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# 4. DATA CLEANING
# ============================================================

st.header("🧹 Data Cleaning")

original_rows = len(df)

# Remove missing customer IDs
df = df.dropna(
    subset=["CustomerID"]
)

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Quantity to number
df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

# Convert UnitPrice to number
df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"],
    errors="coerce"
)

# Remove missing values
df = df.dropna(
    subset=[
        "Quantity",
        "UnitPrice"
    ]
)

# Keep only positive quantity
df = df[
    df["Quantity"] > 0
]

# Keep only positive price
df = df[
    df["UnitPrice"] > 0
]

# Customer ID as string
df["CustomerID"] = (
    df["CustomerID"]
    .astype(str)
)

# ------------------------------------------------------------
# IMPORTANT:
# The Online Retail dataset contains dates like:
# 13-12-2010 09:02
#
# Therefore dayfirst=True is used.
# ------------------------------------------------------------

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    dayfirst=True,
    errors="coerce"
)

# Remove invalid dates
df = df.dropna(
    subset=["InvoiceDate"]
)

# Calculate sales
df["TotalAmount"] = (
    df["Quantity"] *
    df["UnitPrice"]
)

cleaned_rows = len(df)

st.write(
    f"Original rows: **{original_rows:,}**"
)

st.write(
    f"Rows after cleaning: **{cleaned_rows:,}**"
)

st.write(
    f"Rows removed: "
    f"**{original_rows - cleaned_rows:,}**"
)


# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("⚙️ Filters & Controls")

st.sidebar.subheader("🌍 Country Filter")

country_list = sorted(
    df["Country"]
    .dropna()
    .unique()
    .tolist()
)

selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All Countries"] + country_list
)


# ============================================================
# 6. APPLY COUNTRY FILTER
# ============================================================

if selected_country == "All Countries":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["Country"] == selected_country
    ].copy()


# Check filtered data
if len(filtered_df) == 0:

    st.error(
        "No data available for the selected country."
    )

    st.stop()


# ============================================================
# 7. FILTER INFORMATION
# ============================================================

st.header("🔎 Current Analysis")

if selected_country == "All Countries":

    st.info(
        "🌍 Showing analysis for **all countries**."
    )

else:

    st.info(
        f"🌍 Showing analysis for **{selected_country}**."
    )


# ============================================================
# 8. SIDEBAR K-MEANS CONTROLS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🤖 K-Means Controls"
)

number_of_clusters = st.sidebar.slider(
    "Number of Customer Groups",
    min_value=2,
    max_value=6,
    value=3,
    step=1
)


# ============================================================
# 9. BUSINESS KPIs
# ============================================================

st.header("📊 Business Analysis")

total_sales = (
    filtered_df["TotalAmount"].sum()
)

total_customers = (
    filtered_df["CustomerID"].nunique()
)

total_quantity = (
    filtered_df["Quantity"].sum()
)

total_orders = (
    filtered_df["InvoiceNo"].nunique()
)

if total_orders > 0:

    average_order_value = (
        total_sales /
        total_orders
    )

else:

    average_order_value = 0


col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "💷 Total Sales",
        f"£{total_sales:,.2f}"
    )

with col2:

    st.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

with col3:

    st.metric(
        "📦 Quantity Sold",
        f"{total_quantity:,.0f}"
    )

with col4:

    st.metric(
        "🧾 Orders",
        f"{total_orders:,}"
    )

with col5:

    st.metric(
        "💰 Average Order",
        f"£{average_order_value:,.2f}"
    )


# ============================================================
# 10. MONTHLY SALES
# ============================================================

st.subheader("📈 Monthly Sales")

filtered_df["Month"] = (
    filtered_df["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    filtered_df
    .groupby("Month")["TotalAmount"]
    .sum()
    .reset_index()
)

fig_month = px.line(
    monthly_sales,
    x="Month",
    y="TotalAmount",
    markers=True,
    title="Monthly Sales Trend",
    color_discrete_sequence=[
        "#3498DB"
    ]
)

fig_month.update_layout(
    xaxis_title="Month",
    yaxis_title="Sales (£)"
)

st.plotly_chart(
    fig_month,
    use_container_width=True
)


# ============================================================
# 11. TOP PRODUCTS
# ============================================================

st.subheader("🏆 Top 10 Products")

top_products = (
    filtered_df
    .groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index()
)

fig_products = px.bar(
    top_products,
    x="TotalAmount",
    y="Description",
    orientation="h",
    title="Top 10 Products by Sales",
    color="TotalAmount",
    color_continuous_scale="Viridis"
)

fig_products.update_layout(
    xaxis_title="Sales (£)",
    yaxis_title="Product"
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)


# ============================================================
# 12. TOP COUNTRIES
# ============================================================

st.subheader("🌍 Sales by Country")

country_sales = (
    filtered_df
    .groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index()
)

fig_country = px.bar(
    country_sales,
    x="TotalAmount",
    y="Country",
    orientation="h",
    title="Top 10 Countries by Sales",
    color="TotalAmount",
    color_continuous_scale="Plasma"
)

fig_country.update_layout(
    xaxis_title="Sales (£)",
    yaxis_title="Country"
)

st.plotly_chart(
    fig_country,
    use_container_width=True
)


# ============================================================
# 13. CUSTOMER ANALYSIS
# ============================================================

st.header("👥 Customer Analysis")

customer_data = (
    filtered_df
    .groupby("CustomerID")
    .agg(
        TotalSpending=(
            "TotalAmount",
            "sum"
        ),

        TotalQuantity=(
            "Quantity",
            "sum"
        ),

        NumberOfOrders=(
            "InvoiceNo",
            "nunique"
        )
    )
    .reset_index()
)


# ============================================================
# 14. CHECK NUMBER OF CUSTOMERS
# ============================================================

if len(customer_data) < number_of_clusters:

    st.warning(
        "The selected number of clusters is greater "
        "than the number of customers in this country."
    )

    st.info(
        f"Please select a maximum of "
        f"{len(customer_data)} clusters."
    )

    st.stop()


# ============================================================
# 15. CUSTOMER TABLE
# ============================================================

st.subheader("Customer Summary")

st.dataframe(
    customer_data.head(20),
    use_container_width=True
)


# ============================================================
# 16. K-MEANS MODEL
# ============================================================

features = [
    "TotalSpending",
    "TotalQuantity",
    "NumberOfOrders"
]

X = customer_data[
    features
]

# Scale the data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# Create K-Means model
model = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)


# Predict clusters
customer_data["Cluster"] = (
    model.fit_predict(
        X_scaled
    )
)


# ============================================================
# 17. CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    customer_data
    .groupby("Cluster")
    .agg(

        NumberOfCustomers=(
            "CustomerID",
            "count"
        ),

        AverageSpending=(
            "TotalSpending",
            "mean"
        ),

        AverageQuantity=(
            "TotalQuantity",
            "mean"
        ),

        AverageOrders=(
            "NumberOfOrders",
            "mean"
        )
    )
    .round(2)
    .reset_index()
)


# ============================================================
# 18. CREATE CUSTOMER SEGMENTS
# ============================================================

cluster_values = cluster_summary[
    [
        "AverageSpending",
        "AverageQuantity",
        "AverageOrders"
    ]
]

cluster_scaler = StandardScaler()

cluster_scaled = (
    cluster_scaler
    .fit_transform(
        cluster_values
    )
)

cluster_summary["ValueScore"] = (
    cluster_scaled.mean(axis=1)
)


sorted_clusters = (
    cluster_summary
    .sort_values(
        "ValueScore",
        ascending=False
    )["Cluster"]
    .tolist()
)


segment_names = {}


if number_of_clusters == 2:

    segment_names[
        sorted_clusters[0]
    ] = "High-Value Customer"

    segment_names[
        sorted_clusters[1]
    ] = "Low-Value Customer"


elif number_of_clusters == 3:

    segment_names[
        sorted_clusters[0]
    ] = "High-Value Customer"

    segment_names[
        sorted_clusters[1]
    ] = "Regular Customer"

    segment_names[
        sorted_clusters[2]
    ] = "Low-Value Customer"


else:

    for i, cluster in enumerate(
        sorted_clusters
    ):

        if i == 0:

            segment_names[
                cluster
            ] = "High-Value Customer"

        elif i == len(sorted_clusters) - 1:

            segment_names[
                cluster
            ] = "Low-Value Customer"

        else:

            segment_names[
                cluster
            ] = f"Regular Customer {i}"


customer_data["CustomerSegment"] = (
    customer_data["Cluster"]
    .map(segment_names)
)

cluster_summary["CustomerSegment"] = (
    cluster_summary["Cluster"]
    .map(segment_names)
)


# ============================================================
# 19. PREDICTION CONTROLS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🔮 Predict Customer"
)

manual_spending = st.sidebar.number_input(
    "💰 Total Spending (£)",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

manual_quantity = st.sidebar.number_input(
    "📦 Total Quantity",
    min_value=0,
    value=100,
    step=10
)

manual_orders = st.sidebar.number_input(
    "🧾 Number of Orders",
    min_value=1,
    value=5,
    step=1
)


# ============================================================
# 20. CUSTOMER PREDICTION
# ============================================================

st.header("🔮 Customer Prediction")

st.write(
    "Enter customer details using the sidebar. "
    "The K-Means model will automatically predict "
    "the customer's group."
)

manual_customer = pd.DataFrame(
    {
        "TotalSpending": [
            manual_spending
        ],

        "TotalQuantity": [
            manual_quantity
        ],

        "NumberOfOrders": [
            manual_orders
        ]
    }
)


manual_scaled = scaler.transform(
    manual_customer
)


predicted_cluster = (
    model.predict(
        manual_scaled
    )[0]
)

predicted_segment = (
    segment_names[
        predicted_cluster
    ]
)


prediction_col1, prediction_col2 = (
    st.columns(2)
)


with prediction_col1:

    st.metric(
        "Predicted Cluster",
        f"Cluster {predicted_cluster}"
    )


with prediction_col2:

    st.metric(
        "Predicted Segment",
        predicted_segment
    )


# ============================================================
# 21. PREDICTION MESSAGE
# ============================================================

if predicted_segment == "High-Value Customer":

    st.success(
        "⭐ The entered customer belongs to the "
        "**High-Value Customer** group."
    )

    st.write(
        """
        **Suggested actions:**

        • Loyalty rewards  
        • Premium offers  
        • Exclusive products  
        • Personalized recommendations  
        • Early access to new products
        """
    )


elif predicted_segment == "Low-Value Customer":

    st.warning(
        "📢 The entered customer belongs to the "
        "**Low-Value Customer** group."
    )

    st.write(
        """
        **Suggested actions:**

        • Re-engagement campaigns  
        • Promotional offers  
        • Discounts  
        • Product recommendations  
        • Special purchase offers
        """
    )


else:

    st.info(
        "🛍️ The entered customer belongs to the "
        "**Regular Customer** group."
    )

    st.write(
        """
        **Suggested actions:**

        • Cross-selling  
        • Bundle offers  
        • Product recommendations  
        • Loyalty rewards  
        • Seasonal promotions
        """
    )


# ============================================================
# 22. ENTERED CUSTOMER DETAILS
# ============================================================

st.subheader(
    "Your Entered Customer Details"
)

input_data = pd.DataFrame(
    {
        "Feature": [
            "Total Spending",
            "Total Quantity",
            "Number of Orders"
        ],

        "Value": [
            f"£{manual_spending:,.2f}",
            f"{manual_quantity:,}",
            f"{manual_orders:,}"
        ]
    }
)

st.dataframe(
    input_data,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# 23. SIMILAR CUSTOMERS
# ============================================================

st.subheader(
    "👥 Customers in the Predicted Group"
)

similar_customers = customer_data[
    customer_data["Cluster"]
    == predicted_cluster
].copy()

similar_customers = (
    similar_customers
    .sort_values(
        "TotalSpending",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    similar_customers[
        [
            "CustomerID",
            "TotalSpending",
            "TotalQuantity",
            "NumberOfOrders",
            "CustomerSegment"
        ]
    ],
    use_container_width=True
)


# ============================================================
# 24. CLUSTER CUSTOMER COUNT
# ============================================================

st.header(
    "📊 Customer Segmentation Results"
)

cluster_count = (
    customer_data["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "NumberOfCustomers"
]

cluster_count["Segment"] = (
    cluster_count["Cluster"]
    .map(segment_names)
)


fig_cluster = px.bar(
    cluster_count,
    x="Segment",
    y="NumberOfCustomers",
    text="NumberOfCustomers",
    color="Segment",
    title="Customers in Each Segment",
    color_discrete_sequence=[
        "#2E86DE",
        "#E67E22",
        "#27AE60",
        "#8E44AD",
        "#E74C3C",
        "#16A085"
    ]
)

fig_cluster.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Number of Customers"
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)


# ============================================================
# 25. CUSTOMER SCATTER PLOT
# ============================================================

st.subheader(
    "Customer Segmentation Visualization"
)

fig_scatter = px.scatter(
    customer_data,
    x="NumberOfOrders",
    y="TotalSpending",
    color="CustomerSegment",
    size="TotalQuantity",

    hover_data=[
        "CustomerID",
        "TotalSpending",
        "TotalQuantity",
        "NumberOfOrders"
    ],

    title="Customers by Orders and Spending",

    color_discrete_sequence=[
        "#E74C3C",
        "#3498DB",
        "#2ECC71",
        "#9B59B6",
        "#F39C12",
        "#1ABC9C"
    ]
)

fig_scatter.update_layout(
    xaxis_title="Number of Orders",
    yaxis_title="Total Spending (£)"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ============================================================
# 26. AVERAGE SPENDING
# ============================================================

st.subheader(
    "Average Spending by Customer Segment"
)

fig_spending = px.bar(
    cluster_summary,
    x="CustomerSegment",
    y="AverageSpending",
    text="AverageSpending",
    color="CustomerSegment",
    title="Average Spending by Segment",

    color_discrete_sequence=[
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#A29BFE"
    ]
)

fig_spending.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Average Spending (£)"
)

st.plotly_chart(
    fig_spending,
    use_container_width=True
)


# ============================================================
# 27. AVERAGE ORDERS
# ============================================================

st.subheader(
    "Average Orders by Customer Segment"
)

fig_orders = px.bar(
    cluster_summary,
    x="CustomerSegment",
    y="AverageOrders",
    text="AverageOrders",
    color="CustomerSegment",
    title="Average Number of Orders by Segment",

    color_discrete_sequence=[
        "#6C5CE7",
        "#00B894",
        "#FDCB6E",
        "#E17055",
        "#0984E3",
        "#D63031"
    ]
)

fig_orders.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Average Number of Orders"
)

st.plotly_chart(
    fig_orders,
    use_container_width=True
)


# ============================================================
# 28. PIE CHART
# ============================================================

st.subheader(
    "Customer Segment Distribution"
)

segment_distribution = (
    customer_data[
        "CustomerSegment"
    ]
    .value_counts()
    .reset_index()
)

segment_distribution.columns = [
    "CustomerSegment",
    "NumberOfCustomers"
]


fig_pie = px.pie(
    segment_distribution,
    names="CustomerSegment",
    values="NumberOfCustomers",
    title="Customer Segment Distribution",

    color_discrete_sequence=[
        "#FF7675",
        "#74B9FF",
        "#55EFC4",
        "#A29BFE",
        "#FDCB6E",
        "#FD79A8"
    ]
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)


# ============================================================
# 29. CUSTOMER SEGMENT SUMMARY
# ============================================================

st.header(
    "📋 Customer Segment Summary"
)

final_summary = cluster_summary[
    [
        "Cluster",
        "CustomerSegment",
        "NumberOfCustomers",
        "AverageSpending",
        "AverageQuantity",
        "AverageOrders"
    ]
]

st.dataframe(
    final_summary,
    use_container_width=True
)


# ============================================================
# 30. TOP CUSTOMERS
# ============================================================

st.header(
    "🏆 Top 10 Customers"
)

top_customers = (
    customer_data
    .sort_values(
        "TotalSpending",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_customers[
        [
            "CustomerID",
            "TotalSpending",
            "TotalQuantity",
            "NumberOfOrders",
            "CustomerSegment"
        ]
    ],
    use_container_width=True
)


# ============================================================
# 31. ANALYSIS REPORT
# ============================================================

st.header(
    "📄 Analysis Report"
)

st.write(
    "This report is automatically generated from "
    "the currently selected country/filter."
)


# ------------------------------------------------------------
# Report - Overview
# ------------------------------------------------------------

st.subheader("1. 📌 Business Overview")

if selected_country == "All Countries":

    st.write(
        f"""
        The analysis covers **all countries** in the dataset.

        • Total sales: **£{total_sales:,.2f}**
        • Total customers: **{total_customers:,}**
        • Total orders: **{total_orders:,}**
        • Total quantity sold: **{total_quantity:,.0f}**
        • Average order value: **£{average_order_value:,.2f}**
        """
    )

else:

    st.write(
        f"""
        The analysis is focused on **{selected_country}**.

        • Total sales: **£{total_sales:,.2f}**
        • Total customers: **{total_customers:,}**
        • Total orders: **{total_orders:,}**
        • Total quantity sold: **{total_quantity:,.0f}**
        • Average order value: **£{average_order_value:,.2f}**
        """
    )


# ------------------------------------------------------------
# Report - Top Product
# ------------------------------------------------------------

st.subheader("2. 🏆 Product Analysis")

if len(top_products) > 0:

    best_product = (
        top_products.iloc[0]["Description"]
    )

    best_product_sales = (
        top_products.iloc[0]["TotalAmount"]
    )

    st.write(
        f"""
        The highest-selling product in the selected
        data is **{best_product}**.

        Its recorded sales are approximately
        **£{best_product_sales:,.2f}**.
        """
    )


# ------------------------------------------------------------
# Report - Top Country
# ------------------------------------------------------------

st.subheader("3. 🌍 Country Analysis")

if len(country_sales) > 0:

    best_country = (
        country_sales.iloc[0]["Country"]
    )

    best_country_sales = (
        country_sales.iloc[0]["TotalAmount"]
    )

    if selected_country == "All Countries":

        st.write(
            f"""
            Among the countries shown in the analysis,
            **{best_country}** has the highest sales.

            Sales: **£{best_country_sales:,.2f}**
            """
        )

    else:

        st.write(
            f"""
            The current country filter is
            **{selected_country}**.

            Sales for this selected country:
            **£{total_sales:,.2f}**
            """
        )


# ------------------------------------------------------------
# Report - Customer Segmentation
# ------------------------------------------------------------

st.subheader(
    "4. 👥 Customer Segmentation Analysis"
)

highest_segment = (
    cluster_summary
    .sort_values(
        "NumberOfCustomers",
        ascending=False
    )
    .iloc[0]
)

largest_segment_name = (
    highest_segment["CustomerSegment"]
)

largest_segment_count = (
    highest_segment["NumberOfCustomers"]
)

st.write(
    f"""
    The largest customer segment is
    **{largest_segment_name}**.

    It contains approximately
    **{largest_segment_count:,} customers**.
    """
)


# ------------------------------------------------------------
# Report - Highest Spending Segment
# ------------------------------------------------------------

st.subheader(
    "5. 💰 Spending Analysis"
)

highest_spending_segment = (
    cluster_summary
    .sort_values(
        "AverageSpending",
        ascending=False
    )
    .iloc[0]
)

st.write(
    f"""
    The segment with the highest average spending is
    **{highest_spending_segment["CustomerSegment"]}**.

    Average spending:
    **£{highest_spending_segment["AverageSpending"]:,.2f}**
    """
)


# ------------------------------------------------------------
# Report - Prediction
# ------------------------------------------------------------

st.subheader(
    "6. 🔮 Prediction Result"
)

st.write(
    f"""
    Based on the customer details entered in the sidebar:

    • Total spending: **£{manual_spending:,.2f}**
    • Total quantity: **{manual_quantity:,}**
    • Number of orders: **{manual_orders:,}**

    The K-Means model predicts:

    **Cluster:** {predicted_cluster}

    **Customer Segment:** {predicted_segment}
    """
)


# ------------------------------------------------------------
# Report - Business Insights
# ------------------------------------------------------------

st.subheader(
    "7. 💡 Business Insights"
)

if total_customers > 0:

    sales_per_customer = (
        total_sales /
        total_customers
    )

else:

    sales_per_customer = 0


st.write(
    f"""
    ### Key Insights

    **1. Sales Performance**

    The selected data generated approximately
    **£{total_sales:,.2f}** in sales.

    **2. Customer Base**

    There are **{total_customers:,} unique customers**
    in the selected data.

    **3. Order Behaviour**

    Customers generated **{total_orders:,} orders**
    with an average order value of
    **£{average_order_value:,.2f}**.

    **4. Customer Value**

    Average sales generated per customer are approximately
    **£{sales_per_customer:,.2f}**.

    **5. Customer Segmentation**

    K-Means clustering was used to group customers
    according to their spending, quantity purchased
    and number of orders.
    """
)


# ------------------------------------------------------------
# Report - Recommendations
# ------------------------------------------------------------

st.subheader(
    "8. 📌 Business Recommendations"
)

st.write(
    """
    **For High-Value Customers**

    • Provide loyalty rewards  
    • Offer premium products  
    • Provide personalized recommendations  
    • Give early access to new products  

    **For Regular Customers**

    • Use cross-selling strategies  
    • Create bundle offers  
    • Recommend related products  
    • Encourage repeat purchases  

    **For Low-Value Customers**

    • Use re-engagement campaigns  
    • Provide targeted promotions  
    • Offer suitable discounts  
    • Recommend popular products
    """
)


# ============================================================
# 32. FINAL MESSAGE
# ============================================================

st.success(
    "✅ Analysis report generated successfully "
    "inside the Streamlit application."
)
