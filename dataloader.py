import pandas as pd
import datetime 
import numpy as np

def load_data():
    df = pd.read_csv(
    'ChargePoint Data CY20Q4.csv', dtype={'Station Name': str}, 
    parse_dates=['Start Date', 'Total Duration (hh:mm:ss)'], infer_datetime_format=True, low_memory=False)

    # Make a unique id for each row
    df['Id'] = df.index
    # Drop columns that are not needed
    df.drop(
        ['User ID', 'County', 'City', 'Postal Code', 'Driver Postal Code',
        'State/Province', 'Currency', 'EVSE ID', 'Transaction Date (Pacific Time)',
        'GHG Savings (kg)', 'Gasoline Savings (gallons)', 'Org Name',
        'Address 1', 'System S/N', 'Ended By',
        'End Time Zone', 'Start Time Zone', 'Country', 'Model Number'
        ], axis=1, inplace=True)

    df['Total Duration (min)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)']).dt.total_seconds()/60
    df['Charging Time (min)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)']).dt.total_seconds()/60
    df.drop(['Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)'], axis=1, inplace=True)

    df['End Date'] = df['End Date'].apply(convert_to_datetime)

    # Create clusters 
    df['Cluster'] = df['Station Name'].str.split(' ').str[4]
    return df

def convert_to_datetime(serial):
    # There's Excel formatting in the `End Data` column, so we need to convert from Excel serial to datetime
    # https://stackoverflow.com/questions/6706231/fetching-datetime-from-float-in-python
    try:
        return pd.to_datetime(serial)
    except:
        seconds = (float(serial) - 25569) * 86400.0
        return datetime.datetime.utcfromtimestamp(seconds)


def create_count_data(df, interval_length=30, save=False):
    """ Create counts for number of sessions in each interval. The `Period` defines when the period starts and runs until the next period in the dataframe."""
    df_combined = pd.DataFrame()
    for cluster in df['Cluster'].unique():
        # Collect each period that the charging session covers in the 'Period' column. This column will contain a list of datetimes :)
        df.loc[df.Cluster == cluster, 'Period'] = np.array([
            pd.date_range(s, e, freq='T') for s, e in zip(df[df.Cluster == cluster]['Start Date'], df[df.Cluster == cluster]['End Date'])
        ], dtype=object)
        # The the count of 
        res = df[df.Cluster == cluster].explode('Period').groupby(pd.Grouper(key='Period', freq=f'{interval_length}T'))['Id'].nunique()
        # rename energy to sessions
        res.rename('Sessions', inplace=True)
        data = res.to_frame()
        data = data.reset_index()
        data.Period = pd.to_datetime(data.Period)
        data["Cluster"] = cluster
        df_combined = pd.concat([df_combined, data])
    if save:
        df_combined.to_csv('charging_session_count_{interval_length}.csv')

    return df_combined
    
