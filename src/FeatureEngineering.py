# File: src/featureEngineering.py
import numpy as np
import pandas as pd


# Builds the Company A feature dataframe: drops unneeded columns, creates identifiers,
# order complexity, delivery delay/status, and quantity difference/status.
def create_features_company_a(df):
    columns_to_drop = ['Ordered Product', 'End Product', 'Quality']
    df_features = df.drop(columns=columns_to_drop, errors='ignore')

    # Unique Order Identifier
    unique_id = (
        df_features['Order Number'].astype(str) + '-' +
        df_features['Order Position'].astype(str)
    )
    insert_pos = df_features.columns.get_loc('Supplier')
    df_features.insert(insert_pos, 'Unique Order Identifier', unique_id)

    # Unique Product Identifier
    unique_product_id = (
        df_features['Supplier'].astype(str) + '-' +
        df_features['Product Article Number'].astype('Int64').astype(str)
    )
    insert_pos = df_features.columns.get_loc('Product Article Number') + 1
    df_features.insert(insert_pos, 'Unique Product Identifier', unique_product_id)

    # Order Complexity
    order_complexity = df_features.groupby('Order Number')['Order Number'].transform('count')
    insert_pos = df_features.columns.get_loc('Unique Order Identifier') + 1
    df_features.insert(insert_pos, 'Order Complexity', order_complexity)

    # Delivery Delay
    delivery_delay = (
        df_features['Arrival Date'] - df_features['Planned Delivery Date']
    ).dt.days
    insert_pos = df_features.columns.get_loc('Ordered Quantity')
    df_features.insert(insert_pos, 'Delivery Delay', delivery_delay)

    # Delivery Status
    conditions = [
        df_features['Delivery Delay'] < 0,
        df_features['Delivery Delay'] == 0,
        df_features['Delivery Delay'] > 0
    ]
    choices = ['too early', 'on time', 'too late']
    delivery_status = np.select(conditions, choices, default='unknown')
    insert_pos = df_features.columns.get_loc('Ordered Quantity')
    df_features.insert(insert_pos, 'Delivery Status', delivery_status)

    # Quantity Difference and Quantity Status
    quantity_difference = df_features['Delivered Quantity'] - df_features['Ordered Quantity']
    conditions = [
        quantity_difference < 0,
        quantity_difference == 0,
        quantity_difference > 0
    ]
    choices = ['under delivered', 'exact', 'over delivered']
    quantity_status = np.select(conditions, choices, default='unknown')
    df_features['Quantity Difference'] = quantity_difference
    df_features['Quantity Status'] = quantity_status

    # Remove old columns
    df_features = df_features.drop(columns=['Order Position'], errors='ignore')
    df_features = df_features.drop(columns=['Product Article Number'], errors='ignore')
    df_features = df_features.drop(columns=['Order Number'], errors='ignore')

    # Move Unique Order Identifier to the front as the key column
    cols = ['Unique Order Identifier'] + [c for c in df_features.columns if c != 'Unique Order Identifier']
    df_features = df_features[cols]

    return df_features


# Builds the Company B feature dataframe: creates artificial Order Position, drops
# unneeded columns, creates identifiers, order complexity, delivery delay/status,
# and quantity difference/status. Output matches the Company A feature format.
def create_features_company_b(df):
    columns_to_drop = ['Ordered Product', 'Ordered Product - Supplement', 'Original Desired Date',
                        'Supplier Number', 'Original Ordered Quantity']
    df_features = df.drop(columns=columns_to_drop, errors='ignore')

    # Artificial Order Position: sequential count per Order Number, starting at 1
    order_position = df_features.groupby('Order Number').cumcount() + 1
    insert_pos = df_features.columns.get_loc('Order Number') + 1
    df_features.insert(insert_pos, 'Order Position', order_position)

    # Unique Order Identifier
    unique_id = (
        df_features['Order Number'].astype(str) + '-' +
        df_features['Order Position'].astype(str)
    )
    insert_pos = df_features.columns.get_loc('Supplier')
    df_features.insert(insert_pos, 'Unique Order Identifier', unique_id)

    # Unique Product Identifier
    unique_product_id = (
        df_features['Supplier'].astype(str) + '-' +
        df_features['Product Article Number'].astype(str)
    )
    insert_pos = df_features.columns.get_loc('Product Article Number') + 1
    df_features.insert(insert_pos, 'Unique Product Identifier', unique_product_id)

    # Order Complexity
    order_complexity = df_features.groupby('Order Number')['Order Number'].transform('count')
    insert_pos = df_features.columns.get_loc('Unique Order Identifier') + 1
    df_features.insert(insert_pos, 'Order Complexity', order_complexity)

    # Delivery Delay
    df_features['Planned Delivery Date'] = pd.to_datetime(df_features['Planned Delivery Date'])
    df_features['Arrival Date'] = pd.to_datetime(df_features['Arrival Date'])
    delivery_delay = (
        df_features['Arrival Date'] - df_features['Planned Delivery Date']
    ).dt.days
    insert_pos = df_features.columns.get_loc('Ordered Quantity')
    df_features.insert(insert_pos, 'Delivery Delay', delivery_delay)

    # Delivery Status
    conditions = [
        df_features['Delivery Delay'] < 0,
        df_features['Delivery Delay'] == 0,
        df_features['Delivery Delay'] > 0
    ]
    choices = ['too early', 'on time', 'too late']
    delivery_status = np.select(conditions, choices, default='unknown')
    insert_pos = df_features.columns.get_loc('Ordered Quantity')
    df_features.insert(insert_pos, 'Delivery Status', delivery_status)

    # Quantity Difference and Quantity Status
    quantity_difference = df_features['Delivered Quantity'] - df_features['Ordered Quantity']
    conditions = [
        quantity_difference < 0,
        quantity_difference == 0,
        quantity_difference > 0
    ]
    choices = ['under delivered', 'exact', 'over delivered']
    quantity_status = np.select(conditions, choices, default='unknown')
    df_features['Quantity Difference'] = quantity_difference
    df_features['Quantity Status'] = quantity_status

    # Remove old columns
    df_features = df_features.drop(columns=['Order Position'], errors='ignore')
    df_features = df_features.drop(columns=['Product Article Number'], errors='ignore')
    df_features = df_features.drop(columns=['Order Number'], errors='ignore')

    # Move Unique Order Identifier to the front as the key column
    cols = ['Unique Order Identifier'] + [c for c in df_features.columns if c != 'Unique Order Identifier']
    df_features = df_features[cols]

    return df_features


# Extracts numeric date features (year, month, quarter, day, weekday) from
# Planned Delivery Date and Arrival Date, then drops the raw datetime columns.
# Shared by both companies, since their feature dataframes share the same format.
def create_date_encoded_features(df):
    df_encoded = df.copy()
    df_encoded['Planned Delivery Date'] = pd.to_datetime(df_encoded['Planned Delivery Date'])
    df_encoded['Arrival Date'] = pd.to_datetime(df_encoded['Arrival Date'])

    planned_pos = df_encoded.columns.get_loc('Planned Delivery Date') + 1
    planned_features = {
        'Planned Year': df_encoded['Planned Delivery Date'].dt.year,
        'Planned Month': df_encoded['Planned Delivery Date'].dt.month,
        'Planned Quarter': df_encoded['Planned Delivery Date'].dt.quarter,
        'Planned Day': df_encoded['Planned Delivery Date'].dt.day,
        'Planned Weekday': df_encoded['Planned Delivery Date'].dt.dayofweek,
    }
    for i, (name, values) in enumerate(planned_features.items()):
        df_encoded.insert(planned_pos + i, name, values)

    arrival_pos = df_encoded.columns.get_loc('Arrival Date') + 1
    arrival_features = {
        'Arrival Year': df_encoded['Arrival Date'].dt.year,
        'Arrival Month': df_encoded['Arrival Date'].dt.month,
        'Arrival Quarter': df_encoded['Arrival Date'].dt.quarter,
        'Arrival Day': df_encoded['Arrival Date'].dt.day,
        'Arrival Weekday': df_encoded['Arrival Date'].dt.dayofweek,
    }
    for i, (name, values) in enumerate(arrival_features.items()):
        df_encoded.insert(arrival_pos + i, name, values)

    df_encoded = df_encoded.drop(columns=['Planned Delivery Date', 'Arrival Date'])
    return df_encoded
