# File: src/preprocessing.py
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# load function for 3.
def load_procurement(path):
    df = pd.read_csv(path, sep=";")
    for col in ["Planned Delivery Date", "Arrival Date", "Original Desired Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# checks data format fr quantity columns for mixed localization formats and text-based corruption. Prints a report of the findings.
def audit_column_formats(df_dict, columns_to_check):
    """
    Scans specified columns and classifies the data format of every row to 
    identify mixed localization formats and text-based corruption.
    """
    print("--- DATA FORMAT AUDIT REPORT ---")

    # Regex patterns for classification
    us_pattern = re.compile(r'^-?\d+(?:\.\d+)?$')  # e.g., 98.0 or 1500
    eu_pattern = re.compile(
        r'^-?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$')  # e.g., 1.600.800,00
    # Contains letters (e.g., 'frog', 'pcs')
    text_pattern = re.compile(r'[a-zA-Z]')

    for key, df in df_dict.items():
        print(f"\nEvaluating Company {key}:")

        for col in columns_to_check:
            if col not in df.columns:
                continue

            # Convert to string, replace NaN strings with actual NaNs for counting
            s = df[col].astype(str).replace('nan', np.nan).dropna().str.strip()

            # Tally occurrences
            us_count = s.str.match(us_pattern).sum()
            eu_count = s.str.match(eu_pattern).sum()
            text_count = s.str.contains(text_pattern, na=False).sum()
            unclassified = len(s) - (us_count + eu_count + text_count)

            print(f"  Column: '{col}' | Total Valid Rows: {len(s)}")
            print(f"    - US/Standard Numeric (98.0):       {us_count}")
            print(f"    - European Numeric (1.600.800,00):  {eu_count}")
            print(f"    - Corrupt Text ('frog', '10 pcs'):  {text_count}")
            if text_count > 0:
                # Print exactly what the corrupt text is
                garbage_samples = s[s.str.contains(
                    text_pattern, na=False)].unique()[:5]
                print(f"      -> SAMPLES FOUND: {garbage_samples}")

# Removes duplicates and rows missing critical temporal/volumetric data. This is after step 4. .  After cleansing, prints the number of rows removed and remaining for each company.
def perform_structural_cleansing(df_dict):

    cleaned_dict = {}

    # Added 'Product Article number' to the list of required columns
    required_columns = ['Product Article number', 'Planned Delivery Date',
                        'Arrival Date', 'Ordered Quantity', 'Delivered Quantity']
    volume_columns = ['Ordered Quantity', 'Delivered Quantity']

    for key, df in df_dict.items():
        initial_rows = df.shape[0]
        df_clean = df.drop_duplicates()

        # 1. Drop rows with missing values in required columns
        available_columns = [
            col for col in required_columns if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=available_columns)

        # 2. Filter out European Data Formats
        for col in volume_columns:
            if col in df_clean.columns:
                # Cast safely to string for regex scanning
                raw_strings = df_clean[col].astype(str)

                # Flag strings that contain a comma OR have more than one dot
                is_european = raw_strings.str.contains(
                    r',', na=False) | (raw_strings.str.count(r'\.') > 1)

                # Drop flagged rows by keeping only what is NOT (~) European
                df_clean = df_clean[~is_european]

        final_rows = df_clean.shape[0]
        print(
            f"Company {key} Cleansing: Removed {initial_rows - final_rows} invalid rows. Remaining: {final_rows}")
        cleaned_dict[key] = df_clean

    return cleaned_dict

# After initial monovariat plot
# Executes secondary data cleansing by applying domain-specific volumetric and chronological threshold filters mapped to individual companies.
def data_cleansing_2(df_dict):

    cleaned_dict = {}

    for key, df in df_dict.items():
        initial_rows = df.shape[0]
        df_filtered = df.copy()

        # --- SHARED FILTER ---
        # Exclude zero or negative quantities (Addresses Company A Rule 2 & standardizes B)
        if 'Ordered Quantity' in df_filtered.columns and 'Delivered Quantity' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['Ordered Quantity'] > 0) &
                                      (df_filtered['Delivered Quantity'] > 0)]

        # --- COMPANY A SPECIFIC FILTERS ---
        if key == 'A':
            # 1. Delete high quantities (>= 14000)
            if 'Ordered Quantity' in df_filtered.columns and 'Delivered Quantity' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['Ordered Quantity'] < 14000) &
                                          (df_filtered['Delivered Quantity'] < 14000)]

            # (Dates for A appear in order, so no strict chronological filters applied here)

        # --- COMPANY B SPECIFIC FILTERS ---
        elif key == 'B':
            # 1. Restrict Arrival and Planned Dates to between 2010 and today
            if 'Arrival Date' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['Arrival Date'].dt.year >= 2010) &
                                          (df_filtered['Arrival Date'].dt.year <= 2027)]

            if 'Planned Delivery Date' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['Planned Delivery Date'].dt.year >= 2010) &
                                          (df_filtered['Planned Delivery Date'].dt.year <= 2027)]

            # 2. Cap/Filter Quantities at 100,000
            if 'Ordered Quantity' in df_filtered.columns and 'Delivered Quantity' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['Ordered Quantity'] <= 100000) &
                                          (df_filtered['Delivered Quantity'] <= 100000)]

        # --- METRICS & RETURN ---
        final_rows = df_filtered.shape[0]
        dropped_rows = initial_rows - final_rows

        print(
            f"Company {key} Threshold Cleansing: Removed {dropped_rows} anomalous rows. Remaining: {final_rows}")
        cleaned_dict[key] = df_filtered

    return cleaned_dict


#After bivariate plot
# setting boundries on delays and quantity.
def data_clean_3(df_dict):

    cleaned_dict = {}

    for key, df in df_dict.items():
        initial_rows = df.shape[0]
        df_transformed = df.copy()

        # --- COMPANY A SPECIFIC FILTERS () ---
        if key == 'A':
            # 1. Cap Quantity at 250
            if 'Ordered Quantity' in df_transformed.columns:
                df_transformed['Ordered Quantity'] = df_transformed['Ordered Quantity'].clip(
                    upper=250)

        # --- COMPANY B SPECIFIC FILTERS () ---
        elif key == 'B':
            # 1. Cap Quantity at 10,000
            if 'Ordered Quantity' in df_transformed.columns:
                df_transformed['Ordered Quantity'] = df_transformed['Ordered Quantity'].clip(
                    upper=10000)

            # 2. Cap Delivery Delta between -30 and +400 days
            if 'Delivery_Delay_Days' in df_transformed.columns:
                df_transformed['Delivery_Delay_Days'] = df_transformed['Delivery_Delay_Days'].clip(
                    lower=-30, upper=400)

        # --- METRICS & RETURN ---
        final_rows = df_transformed.shape[0]

        # Validation print matching your style (Note: Winsorization does not drop rows)
        print(
            f"Company {key}: Capped extreme values. Total rows maintained: {final_rows}")

        cleaned_dict[key] = df_transformed

    return cleaned_dict

def get_supplier_metrics(df_dict, company_key, min_orders=30):
    """
    Calculates delivery status and aggregates average/median delays 
    and fulfillment percentages per supplier.
    """
    df = df_dict[company_key].copy()

    # Ensure delay exists
    if 'Delivery_Delay_Days' not in df.columns:
        df['Delivery_Delay_Days'] = (
            df['Arrival Date'] - df['Planned Delivery Date']).dt.days

    # Assign Delivery Status
    def assign_status(d):
        if pd.isna(d):
            return "open"
        if d > 0:
            return "late"
        if d < 0:
            return "early"
        return "on_time"

    df['Status'] = df['Delivery_Delay_Days'].apply(assign_status)

    # Filter for completed deliveries to assess performance
    delivered = df[df['Status'] != 'open']

    # Aggregate by Supplier
    if 'Supplier' in delivered.columns or 'Supplier Number' in delivered.columns:
        supplier_col = 'Supplier' if 'Supplier' in delivered.columns else 'Supplier Number'

        g = delivered.groupby(supplier_col)
        summary = pd.DataFrame({
            "Total_Orders": g.size(),
            "Mean_Delay":   g["Delivery_Delay_Days"].mean().round(2),
            "Median_Delay": g["Delivery_Delay_Days"].median(),
            "Late_%":       (g["Status"].apply(lambda s: (s == "late").mean()) * 100).round(1),
            "On_Time_%":    (g["Status"].apply(lambda s: (s == "on_time").mean()) * 100).round(1),
            "Early_%":      (g["Status"].apply(lambda s: (s == "early").mean()) * 100).round(1),
        })

        # Filter out low-volume suppliers for statistical significance
        summary = summary[summary["Total_Orders"] >= min_orders]
        summary = summary.sort_values("Late_%", ascending=False)

        print(
            f"--- Company {company_key}: Top Suppliers by Late Percentage ---")
        display(summary.head(10))
        return df, summary
    else:
        print(f"No Supplier column found for Company {company_key}")
        return df, None


def analyze_quantity_delay_correlation(df_dict, company_key):
    """
    Calculates the correlation matrix between Ordered Quantity, 
    Delivered Quantity, and Delivery Delay.
    """
    df = df_dict[company_key].dropna(subset=['Arrival Date']).copy()

    if 'Delivery_Delay_Days' not in df.columns:
        df['Delivery_Delay_Days'] = (
            df['Arrival Date'] - df['Planned Delivery Date']).dt.days

    cols = ['Ordered Quantity', 'Delivered Quantity', 'Delivery_Delay_Days']

    # Verify columns exist
    available_cols = [col for col in cols if col in df.columns]

    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        print(f"--- Company {company_key}: Correlation Matrix ---")
        display(corr_matrix.round(3))

        if 'Ordered Quantity' in available_cols and 'Delivery_Delay_Days' in available_cols:
            q_d_corr = corr_matrix.loc['Ordered Quantity',
                                       'Delivery_Delay_Days']
            if abs(q_d_corr) < 0.1:
                print(
                    "Insight: Order size has virtually no linear impact on lateness.\n")
            elif q_d_corr > 0.1:
                print("Insight: Larger orders correlate slightly with longer delays.\n")
            else:
                print(
                    "Insight: Larger orders correlate slightly with earlier deliveries.\n")


def analyze_order_complexity(df_dict, company_key):
    """
    Groups data by Order Number to count the number of positions (lines) per order
    and correlates this complexity with the maximum delay of that order.
    """
    df = df_dict[company_key].dropna(subset=['Arrival Date']).copy()

    if 'Order Number' not in df.columns:
        print(f"Company {company_key} lacks an 'Order Number' column.\n")
        return

    if 'Delivery_Delay_Days' not in df.columns:
        df['Delivery_Delay_Days'] = (
            df['Arrival Date'] - df['Planned Delivery Date']).dt.days

    # Group by order number to find complexity (row count) and max delay
    order_group = df.groupby('Order Number').agg(
        Order_Positions=('Order Number', 'count'),
        Max_Delay_Days=('Delivery_Delay_Days', 'max')
    )

    # Correlate the number of positions with the delay
    complexity_corr = order_group['Order_Positions'].corr(
        order_group['Max_Delay_Days'])

    print(f"--- Company {company_key}: Order Complexity Analysis ---")
    print(
        f"Average positions per order: {order_group['Order_Positions'].mean():.2f}")
    print(f"Correlation (Positions vs. Max Delay): {complexity_corr:.3f}")

    # Visualize the distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(order_group['Order_Positions'],
               order_group['Max_Delay_Days'], alpha=0.3, color='teal')
    ax.set_title(f'Company {company_key}: Order Positions vs Max Delay')
    ax.set_xlabel('Number of Positions in Order')
    ax.set_ylabel('Maximum Delay (Days)')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()


def visualize_comprehensive_correlations(df_dict, company_key):
    """
    Creates a correlation heatmap and a scatter plot matrix to visualize 
    the relationships between Quantity, Delay, and Order Complexity 
    using ONLY matplotlib and pandas.
    """
    df = df_dict[company_key].dropna(subset=['Arrival Date']).copy()

    if 'Order Number' not in df.columns:
        print(
            f"Company {company_key} lacks an 'Order Number' column for aggregation.\n")
        return

    if 'Delivery_Delay_Days' not in df.columns:
        df['Delivery_Delay_Days'] = (
            df['Arrival Date'] - df['Planned Delivery Date']).dt.days

    # 1. Aggregate at the Order Level to compare all three metrics cleanly
    order_level = df.groupby('Order Number').agg(
        Order_Positions=('Order Number', 'count'),
        Total_Ordered_Quantity=('Ordered Quantity', 'sum'),
        Max_Delay_Days=('Delivery_Delay_Days', 'max')
    )

    print(
        f"========== Company {company_key}: Multivariate Visualization ==========")

    # 2. Correlation Heatmap (Pure Matplotlib)
    corr_matrix = order_level.corr()
    fig, ax = plt.subplots(figsize=(7, 5))

    # Create the color mesh
    cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, label='Pearson Correlation')

    # Set labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=15, ha='right')
    ax.set_yticklabels(corr_matrix.columns)

    # Annotate the squares with the exact correlation numbers
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center',
                    va='center', color=text_color)

    plt.title(f'Company {company_key}: Correlation Heatmap', pad=15)
    plt.tight_layout()
    plt.show()

   # 3. Scatter Plot Matrix (Pure Pandas)
    # Samples the data if it exceeds 2000 orders to prevent Jupyter from freezing
    plot_data = order_level.sample(n=min(2000, len(
        order_level)), random_state=42) if len(order_level) > 2000 else order_level

    # Removed 'edgecolor' to prevent the pandas/matplotlib alias conflict
    axes = pd.plotting.scatter_matrix(
        plot_data,
        alpha=0.4,
        figsize=(9, 9),
        diagonal='hist',
        color='steelblue'
    )

    plt.suptitle(
        f'Company {company_key}: Scatter Plot Matrix (Order Level)', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_temporal_trends(df_dict, company_key):
    """
    Analyzes delivery delays based on the Day of the Week and Month
    the delivery was PLANNED to arrive, displaying the late percentage.
    """
    df = df_dict[company_key].dropna(
        subset=['Arrival Date', 'Planned Delivery Date']).copy()

    if 'Delivery_Delay_Days' not in df.columns:
        df['Delivery_Delay_Days'] = (
            df['Arrival Date'] - df['Planned Delivery Date']).dt.days

    # 1. Extract Temporal Features from the PLANNED date
    df['Planned_DayOfWeek'] = df['Planned Delivery Date'].dt.dayofweek  # 0=Mon, 6=Sun
    df['Planned_Month'] = df['Planned Delivery Date'].dt.month         # 1=Jan, 12=Dec

    # 2. Aggregate Data (Calculate Late Percentage)
    # Using reindex to ensure missing days/months still show up as 0 on the chart
    day_stats = df.groupby('Planned_DayOfWeek').agg(
        Total_Orders=('Delivery_Delay_Days', 'count'),
        Late_Pct=('Delivery_Delay_Days', lambda x: (x > 0).mean() * 100)
    ).reindex(range(7), fill_value=0)

    month_stats = df.groupby('Planned_Month').agg(
        Total_Orders=('Delivery_Delay_Days', 'count'),
        Late_Pct=('Delivery_Delay_Days', lambda x: (x > 0).mean() * 100)
    ).reindex(range(1, 13), fill_value=0)

    # Day/Month Labels for the charts
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(
        f"========== Company {company_key}: Temporal & Seasonal Context ==========")

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Chart 1: Day of Week ---
    axes[0].bar(days, day_stats['Late_Pct'], color='steelblue',
                edgecolor='black', alpha=0.8)
    axes[0].set_title('Late Delivery Risk by Day of the Week')
    axes[0].set_ylabel('Percentage of Orders Late (%)')

    # Add a baseline average line for easy visual comparison
    avg_day_late = df['Delivery_Delay_Days'].apply(
        lambda x: x > 0).mean() * 100
    axes[0].axhline(avg_day_late, color='red', linestyle='--',
                    linewidth=1.5, label=f'Overall Avg ({avg_day_late:.1f}%)')
    axes[0].legend()

    # --- Chart 2: Month ---
    axes[1].bar(months, month_stats['Late_Pct'],
                color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[1].set_title('Late Delivery Risk by Month (Seasonality)')
    axes[1].set_ylabel('Percentage of Orders Late (%)')

    axes[1].axhline(avg_day_late, color='red', linestyle='--',
                    linewidth=1.5, label=f'Overall Avg ({avg_day_late:.1f}%)')
    axes

