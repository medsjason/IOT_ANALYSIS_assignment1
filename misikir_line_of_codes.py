
# for dataset 2025-03-01
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-01.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
# Adjust filename as needed
df = pd.read_csv("sensor_data_2025-03-01.csv")

# Expected columns: "timestamp", "temperature", "humidity", "light intensity"
df.columns = [col.lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + HOURLY GROUPING
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.floor('h') # round to hour

# Hourly average 
hourly = df.groupby('hour').agg({
    'temperature': 'mean',
    'humidity': 'mean',
    'light': 'mean'
}).reset_index()

# 3. FIND MAX & MIN VALUES AND TIMES
max_temp_time = hourly.loc[hourly['temperature'].idxmax(), 'hour']
min_temp_time = hourly.loc[hourly['temperature'].idxmin(), 'hour']

max_hum_time = hourly.loc[hourly['humidity'].idxmax(), 'hour']
min_hum_time = hourly.loc[hourly['humidity'].idxmin(), 'hour']

max_light_time = hourly.loc[hourly['light'].idxmax(), 'hour']
min_light_time = hourly.loc[hourly['light'].idxmin(), 'hour']

# PRINT RESULTS
print("\n Daily Max/Min Values")
print(f"Max Temperature at: {max_temp_time}")
print(f"Min Temperature at: {min_temp_time}")
print(f"Max Humidity at: {max_hum_time}")
print(f"Min Humidity at: {min_hum_time}")
print(f"Max Light Intensity at: {max_light_time}")
print(f"Min Light Intensity at: {min_light_time}")
print("\n")

# 4. PLOTS (HOURLY TRENDS)
# Temperature Trend
plt.figure(figsize=(12, 5))
plt.plot(hourly['hour'], hourly['temperature'], marker='o')
plt.title("Temperature vs Time (Hourly)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

# Humidity Trend
plt.figure(figsize=(12, 5))
plt.plot(hourly['hour'], hourly['humidity'], marker='o', color='green')
plt.title("Humidity vs Time (Hourly)")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Light Intensity Trend
plt.figure(figsize=(12, 5))
plt.plot(hourly['hour'], hourly['light'], marker='o', color='orange')
plt.title("Light Intensity vs Time (Hourly)")
plt.ylabel("Light Intensity")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-01.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()
# 1.2 Patterns

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-01.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. Scatter plot to visualize relationship between humidity and temperature
plt.figure(figsize=(12, 5))

plt.scatter(df['temperature'], df['humidity'], 
            alpha=0.6, s=40, edgecolor='black',
            c=df['temperature'], cmap='coolwarm', label="Data")

# Add trendline
z = np.polyfit(df['temperature'], df['humidity'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'], p(df['temperature']), 
         color='green', linewidth=2, label='Trendline')

plt.title("Scatter Plot: Temperature vs Humidity with Trendline")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 8. Scatter plot to visualize relationship between light and temperature
plt.figure(figsize=(12, 5))

plt.scatter(df['temperature'], df['light'], 
            alpha=0.6, s=40, edgecolor='black',
            c=df['temperature'], cmap='coolwarm', label="Data")

# Add trendline
z = np.polyfit(df['temperature'], df['light'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'], p(df['temperature']), 
         color='green', linewidth=2, label='Trendline')

plt.title("Scatter Plot: Temperature vs light with Trendline")
plt.xlabel("Temperature (°C)")
plt.ylabel("light (lux)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 9. Scatter plot to visualize relationship between light and humidity
plt.figure(figsize=(12, 5))

plt.scatter(df['light'], df['humidity'], 
            alpha=0.6, s=40, edgecolor='black',
            c=df['light'], cmap='coolwarm', label="Data")

# Add trendline
z = np.polyfit(df['light'], df['humidity'], 1)
p = np.poly1d(z)
plt.plot(df['light'], p(df['light']), 
         color='green', linewidth=2, label='Trendline')

plt.title("Scatter Plot: humidity vs light with Trendline")
plt.xlabel("light (lux)")
plt.ylabel("humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.3 Basic stastics
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-01.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# (d) TIME-SERIES PLOTS WITH MAX/MIN INDICATORS

for sensor in sensor_cols:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[sensor], label=f"{sensor.capitalize()} Over Time", linewidth=1.5)

    # Identify max and min points
    max_val = df[sensor].max()
    min_val = df[sensor].min()
    max_time = df[sensor].idxmax()
    min_time = df[sensor].idxmin()

    # Plot max/min points on graph
    plt.scatter(max_time, max_val, color='red', s=60, label=f"Max: {max_val:.2f}")
    plt.scatter(min_time, min_val, color='blue', s=60, label=f"Min: {min_val:.2f}")

    # Annotate max
    plt.annotate(f"Max\n{max_val:.2f}\n{max_time}",
                 xy=(max_time, max_val),
                 xytext=(max_time, max_val + (0.05 * max_val)),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=9)

    # Annotate min
    plt.annotate(f"Min\n{min_val:.2f}\n{min_time}",
                 xy=(min_time, min_val),
                 xytext=(min_time, min_val + (0.05 * max_val)),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=9)

    # Labels
    plt.title(f"{sensor.capitalize()} Over Time with Max & Min Indicators")
    plt.xlabel("Time")
    plt.ylabel(sensor.capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-01.csv")
df.columns = [col.strip().lower() for col in df.columns]

# 2. ROLLING MEAN & STD
window_size = 50  # Adjust based on resolution
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

# 3. DETECT ANOMALIES
# Temperature & humidity anomalies (optional)
for col in ['temperature', 'humidity']:
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# Light intensity anomalies (detect artificial indoor lights)
# Threshold can be based on rolling mean + 2 std or absolute intensity
light_threshold = df['light_rolling_mean'] + 2 * df['light_rolling_std']
df['light_anomaly'] = df['light'] > light_threshold
# 4. PLOT LIGHT ANOMALIES (Artificial Light Detection)
plt.figure(figsize=(12,5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']],
            df['light'][df['light_anomaly']],
            color='yellow', edgecolors='black', label='Artificial Light Detected', zorder=5)
plt.title(f"Indoor Artificial Light Detection (Count: {df['light_anomaly'].sum()})")
plt.xlabel("Index / Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 5. SUMMARY
print(f"Total detected artificial light events: {df['light_anomaly'].sum()}")

# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-01.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-01.csv")

# Select numeric sensor columns
sensor_cols = df.select_dtypes(include=np.number).columns
data = df[sensor_cols]

print("Numeric sensor columns:", list(sensor_cols))

# 2. SENSOR FAULT DETECTIONS
faults = pd.DataFrame(index=df.index)

# 2.1 Missing sensor readings
faults["missing_values"] = data.isna().any(axis=1)

# 2.2 Stuck sensor detection (rolling window std == 0)
window = 50
faults["stuck_sensor"] = (
    data.rolling(window=window)
        .std()
        .fillna(0)
        .eq(0)
        .any(axis=1)
)

# 2.3 Spike / sudden jump detection
diff = data.diff().abs()
spike_matrix = diff > (diff.mean() + 2 * diff.std())
faults["spike_detected"] = spike_matrix.any(axis=1)

# 2.4 OUT-OF-RANGE VALUES
sensor_ranges = {
    "temperature": (20, 25),
    "light intensity": (100, 1000),
    "humidity": (40, 60)
}

for col in sensor_ranges:
    if col in data.columns:
        min_val, max_val = sensor_ranges[col]
        faults[f"{col}_out_of_range"] = (data[col] < min_val) | (data[col] > max_val)

# 3. ML-BASED ANOMALY DETECTION
clf = IsolationForest(contamination=0.05, random_state=42)
faults["ml_anomaly"] = clf.fit_predict(data.fillna(0))
faults["ml_anomaly"] = faults["ml_anomaly"].map({1: False, -1: True})

# 4. COMBINE ALL ANOMALIEy
faults["any_fault"] = faults.any(axis=1)

print("\n=== Summary of detected faults ===")
print(faults["any_fault"].value_counts())

faults.to_csv("sensor_fault_detection_results.csv", index=False)
print("\nResults saved to 'sensor_fault_detection_results.csv'")

# 5. VISUALIZATION (FIXED)

plt.figure(figsize=(12, 5))

# First sensor column
sensor_name = data.columns[0]

plt.plot(data.index, data[sensor_name], label=f"{sensor_name}")

# FIX: Use .loc instead of iloc for boolean masking
plt.scatter(
    faults.index[faults["any_fault"]],
    data.loc[faults["any_fault"], sensor_name],
    color="red",
    marker="o",
    label="Detected Fault/Outlier"
)

plt.legend()
plt.title("Sensor Fault & Outlier Detection")
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-01.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

# 3. SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 3. FEATURE VISUALIZATION
# 3.1 Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3.2 Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3.3 Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
# for dataset 2025-03-02(day 2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-02.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')
# For dataset 2025-03-02
# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-02.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()

# Temperature versus humidity
plt.figure(figsize=(7, 5))

# Temperature points (x-axis) in red
plt.scatter(df['temperature'], df['humidity'], 
            alpha=0.6, color='red', label='Temperature')

# Humidity points (y-axis) in blue
plt.scatter(df['temperature'], df['humidity'], 
            alpha=0.6, color='blue', label='Humidity')

# Trendline
z = np.polyfit(df['temperature'], df['humidity'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'], p(df['temperature']), color='black', linewidth=2)

plt.title("Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Temerature versus light
plt.figure(figsize=(7, 5))
# Temperature points in red
plt.scatter(df['temperature'], df['light'], 
            alpha=0.6, color='red', label='Temperature')

# Light points in green
plt.scatter(df['temperature'], df['light'], 
            alpha=0.6, color='green', label='Light Intensity')

# Trendline
z = np.polyfit(df['temperature'], df['light'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'], p(df['temperature']), color='black', linewidth=2)

plt.title("Temperature vs Light Intensity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Light Intensity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Humidity versus light
plt.figure(figsize=(7, 5))
# Humidity points in blue
plt.scatter(df['humidity'], df['light'], 
            alpha=0.6, color='blue', label='Humidity')

# Light points in green
plt.scatter(df['humidity'], df['light'], 
            alpha=0.6, color='green', label='Light Intensity')

# Trendline
z = np.polyfit(df['humidity'], df['light'], 1)
p = np.poly1d(z)
plt.plot(df['humidity'], p(df['humidity']), color='black', linewidth=2)

plt.title("Humidity vs Light Intensity")
plt.xlabel("Humidity (%)")
plt.ylabel("Light Intensity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-02.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. Scatter plot to visualize relationship between humidity and temperature
plt.figure(figsize=(12, 5))

plt.scatter(df['temperature'], df['humidity'], 
            alpha=0.6, s=40, edgecolor='black',
            c=df['temperature'], cmap='coolwarm', label="Data")

# Add trendline
z = np.polyfit(df['temperature'], df['humidity'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'], p(df['temperature']), 
         color='green', linewidth=2, label='Trendline')

plt.title("Scatter Plot: Temperature vs Humidity with Trendline")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 8. Scatter plot to visualize relationship between light and temperature
plt.figure(figsize=(12, 5))

plt.scatter(df['temperature'], df['light'], 
            alpha=0.6, s=40, edgecolor='black',
            c=df['temperature'], cmap='coolwarm', label="Data")

# Add trendline
z = np.polyfit(df['temperature'], df['light'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'], p(df['temperature']), 
         color='green', linewidth=2, label='Trendline')

plt.title("Scatter Plot: Temperature vs light with Trendline")
plt.xlabel("Temperature (°C)")
plt.ylabel("light (lux)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 9. Scatter plot to visualize relationship between light and humidity
plt.figure(figsize=(12, 5))

plt.scatter(df['light'], df['humidity'], 
            alpha=0.6, s=40, edgecolor='black',
            c=df['light'], cmap='coolwarm', label="Data")

# Add trendline
z = np.polyfit(df['light'], df['humidity'], 1)
p = np.poly1d(z)
plt.plot(df['light'], p(df['light']), 
         color='green', linewidth=2, label='Trendline')

plt.title("Scatter Plot: humidity vs light with Trendline")
plt.xlabel("light (lux)")
plt.ylabel("humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-02.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-02.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-02.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

#  SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
# for dataset 2025-03-03(day 3)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-03.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')
# For dataset 2025-03-03
# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-03.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()
# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-03.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. SCATTER PLOT TO VISUALIZE INVERSE RELATION
plt.figure(figsize=(12, 5))
plt.scatter(df['temperature'], df['humidity'], alpha=0.5, color='purple', label='Temp vs Humidity')
plt.title("Scatter Plot: Temperature vs Humidity")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.legend()
plt.show()


# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-03.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-03.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-03.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

#  SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
# for dataset 2025-03-04(day 4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-04.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')

# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-04.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()
# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-04.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. SCATTER PLOT TO VISUALIZE INVERSE RELATION
plt.figure(figsize=(12, 5))
plt.scatter(df['temperature'], df['humidity'], alpha=0.5, color='purple', label='Data Points')
plt.title("Scatter Plot: Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-04.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-04.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-04.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

#  SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
# for dataset 2025-03-05(day 5)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-05.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')

# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-05.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()
# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-05.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. SCATTER PLOT TO VISUALIZE INVERSE RELATION
plt.figure(figsize=(12, 5))
plt.scatter(df['temperature'], df['humidity'], alpha=0.5, color='purple', label='Data Points')
plt.title("Scatter Plot: Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-05.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-05.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-05.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

#  SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
# for dataset 2025-03-06(day 6)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-06.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')
# For dataset 2025-03-06
# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-06.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()
# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-06.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. SCATTER PLOT TO VISUALIZE INVERSE RELATION
plt.figure(figsize=(12, 5))
plt.scatter(df['temperature'], df['humidity'], alpha=0.5, color='purple', label='Data Points')
plt.title("Scatter Plot: Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-06.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-06.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-06.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

#  SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
# for dataset 2025-03-07(day 7)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET
df = pd.read_csv("sensor_data_2025-03-07.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. CONVERT TIMESTAMP TO DATETIME + MINUTE GROUPING
time_col = next((c for c in ['timestamp', 'time', 'datetime', 'date'] if c in df.columns), None)
if time_col is None:
    raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

# Convert timestamp to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])

# Create 'minute' column (rounding down to the nearest minute)
df['minute'] = df[time_col].dt.floor('min')  # 'min' = minute frequency

# 3. CHECK SENSOR COLUMNS
required_cols = ['temperature', 'humidity', 'light']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 4. CALCULATE MINUTE-AVERAGE VALUES
minute_avg = df.groupby('minute')[required_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'minute'], df.loc[df[col].idxmin(), 'minute']

max_temp, min_temp = get_max_min_times(minute_avg, 'temperature')
max_hum, min_hum = get_max_min_times(minute_avg, 'humidity')
max_light, min_light = get_max_min_times(minute_avg, 'light')

print("\n DAILY MAX/MIN VALUES (MINUTE AVERAGE)")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 6. PLOTTING MINUTE AVERAGE TRENDS
def plot_trend(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['minute'], df[col], marker='o', linestyle='-', color=color)
    plt.title(f"{ylabel} vs Time (Minute Average)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trend(minute_avg, 'temperature', 'Temperature', color='red')
plot_trend(minute_avg, 'humidity', 'Humidity (%)', color='green')
plot_trend(minute_avg, 'light', 'Light Intensity', color='orange')
# For dataset 2025-03-07
# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sensor_data_2025-03-07.csv")
# Convert timestamp coumn to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()
# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-07.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 250
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. SCATTER PLOT TO VISUALIZE INVERSE RELATION
plt.figure(figsize=(12, 5))
plt.scatter(df['temperature'], df['humidity'], alpha=0.5, color='purple', label='Data Points')
plt.title("Scatter Plot: Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Load Dataset
df = pd.read_csv("sensor_data_2025-03-07.csv")

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# (d) TIME-SERIES PLOTS WITH MAX/MIN INDICATORS

for sensor in sensor_cols:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[sensor], label=f"{sensor.capitalize()} Over Time", linewidth=1.5)

    # Identify max and min points
    max_val = df[sensor].max()
    min_val = df[sensor].min()
    max_time = df[sensor].idxmax()
    min_time = df[sensor].idxmin()

    # Plot max/min points on graph
    plt.scatter(max_time, max_val, color='red', s=60, label=f"Max: {max_val:.2f}")
    plt.scatter(min_time, min_val, color='blue', s=60, label=f"Min: {min_val:.2f}")

    # Annotate max
    plt.annotate(f"Max\n{max_val:.2f}\n{max_time}",
                 xy=(max_time, max_val),
                 xytext=(max_time, max_val + (0.05 * max_val)),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=9)

    # Annotate min
    plt.annotate(f"Min\n{min_val:.2f}\n{min_time}",
                 xy=(min_time, min_val),
                 xytext=(min_time, min_val + (0.05 * max_val)),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=9)

    # Labels
    plt.title(f"{sensor.capitalize()} Over Time with Max & Min Indicators")
    plt.xlabel("Time")
    plt.ylabel(sensor.capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset 
df = pd.read_csv("sensor_data_2025-03-07.csv")
# Define a rolling window for the trend line
window_size = 50  # Adjust based on your data resolution
df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['hum_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
df['light_rolling_mean'] = df['light'].rolling(window=window_size).mean()
# Compute rolling mean and std
window_size = 50
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
 
# Define anomaly: if current value deviates > 2 * std from rolling mean
    df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])

# 1️ TEMPERATURE ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature'], color='gray', alpha=0.6, label='Temperature')
plt.plot(df.index, df['temp_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['temperature_anomaly']], 
            df['temperature'][df['temperature_anomaly']], 
            color='red', label='Anomalies', zorder=5)

# Count anomalies
temp_anomaly_count = df['temperature_anomaly'].sum()

plt.title(f"Temperature Anomaly Detection (Count: {temp_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2️ HUMIDITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['humidity'], color='lightblue', alpha=0.6, label='Humidity')
plt.plot(df.index, df['hum_rolling_mean'], color='red', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['humidity_anomaly']],
            df['humidity'][df['humidity_anomaly']],
            color='blue', label='Anomalies', zorder=5)

hum_anomaly_count = df['humidity_anomaly'].sum()

plt.title(f"Humidity Anomaly Detection (Count: {hum_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3️ LIGHT INTENSITY ANOMALIES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='green', alpha=0.6, label='Light Intensity')
plt.plot(df.index, df['light_rolling_mean'], color='orange', linewidth=2, label='Rolling Mean')
plt.scatter(df.index[df['light_anomaly']], 
            df['light'][df['light_anomaly']], 
            color='yellow', edgecolors='black', label='Anomalies', zorder=5)

light_anomaly_count = df['light_anomaly'].sum()

plt.title(f"Light Intensity Anomaly Detection (Count: {light_anomaly_count})")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4️  SUMMARY OF ANOMALY COUNTS
print("\n=== Total Anomaly Counts ===")
print(f"Temperature anomalies: {temp_anomaly_count}")
print(f"Humidity anomalies: {hum_anomaly_count}")
print(f"Light intensity anomalies: {light_anomaly_count}")

plt.figure(figsize=(6,4))
plt.bar(['Temperature', 'Humidity', 'Light'],
        [temp_anomaly_count, hum_anomaly_count, light_anomaly_count],
        color=['red', 'blue', 'green'])
plt.title("Anomaly Distribution per Sensor")
plt.ylabel("Count of Anomalies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("sensor_data_2025-03-07.csv")
print(df.columns.tolist())


# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely with min_periods=1 to reduce NaNs
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    # Rolling mean and std with min_periods=1 (prevents NaN buildup)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    # Flag anomaly: deviation > 2 ×std from rolling mean
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# 2.4 Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# 2.5 Handle NaN values explicitly (fill with mean of valid entries)
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# 2.6 Evaluate NaN values after filling
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

#  SAVE ENGINEERED DATA
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET & FIX BAD HEADERS

# Load raw file
raw = pd.read_csv("sensor_data_weekly.csv", header=None)

# Find the first row that looks like column names
header_row = None
for i in range(5):  # check first 5 rows only
    row = raw.iloc[i].astype(str)
    if any(keyword in " ".join(row).lower() 
           for keyword in ["time", "timestamp", "date"]):
        header_row = i
        break

if header_row is None:
    raise ValueError("Could not find a valid header row with timestamp/time/date.")

# Reload using that row as header
df = pd.read_csv("sensor_data_weekly.csv", header=header_row)

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. FIND & FIX TIMESTAMP COLUMN

possible_time_cols = ['timestamp', 'time', 'datetime', 'date']
time_col = next((c for c in possible_time_cols if c in df.columns), None)

if time_col is None:
    raise KeyError(f" Timestamp column NOT FOUND. Columns detected: {df.columns.tolist()}")

df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])  # remove broken timestamps
df['hour'] = df[time_col].dt.floor('h')

# 3. CHECK SENSOR COLUMNS
required = ["temperature", "humidity", "light"]
for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing column: '{col}' — available: {df.columns.tolist()}")

# 4. HOURLY AVERAGES
hourly = df.groupby("hour")[required].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min(df, col):
    return df.loc[df[col].idxmax(), "hour"], df.loc[df[col].idxmin(), "hour"]

maxT, minT = get_max_min(hourly, "temperature")
maxH, minH = get_max_min(hourly, "humidity")
maxL, minL = get_max_min(hourly, "light")

print("\n DAILY MAX/MIN TIMES")
print(f" Max Temperature at: {maxT}")
print(f" Min Temperature at: {minT}")
print(f" Max Humidity at: {maxH}")
print(f" Min Humidity at: {minH}")
print(f" Max Light at: {maxL}")
print(f" Min Light at: {minL}\n")

# 6. PLOTTING
def plot_hourly(df, col, ylabel, color=None):
    plt.figure(figsize=(12,5))
    plt.plot(df['hour'], df[col], marker='o', color=color)
    plt.title(f"{ylabel} (Hourly)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_hourly(hourly, "temperature", "Temperature", color="red")
plot_hourly(hourly, "humidity", "Humidity (%)", color="green")
plot_hourly(hourly, "light", "Light Intensity", color="orange")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATASET (FIX BROKEN HEADERS)

# Load raw CSV 
raw = pd.read_csv("sensor_data_weekly.csv", header=None)

# Automatically detect the row containing real column names
header_row = None
for i in range(10):  # check first 10 rows
    row_text = " ".join(raw.iloc[i].astype(str).str.lower())
    if any(x in row_text for x in ["time", "timestamp", "date"]):
        header_row = i
        break

if header_row is None:
    raise ValueError(" Could not find a header row with timestamp/time/date.")

# Reload with correct header
df = pd.read_csv("sensor_data_weekly.csv", header=header_row)

# Clean column names
df.columns = [col.strip().lower() for col in df.columns]

# 2. DETECT REAL TIMESTAMP COLUMN
possible_time_cols = ['timestamp', 'time', 'datetime', 'date']
time_col = next((c for c in possible_time_cols if c in df.columns), None)

if time_col is None:
    raise KeyError(
        f" No timestamp column found. Columns detected: {list(df.columns)}"
    )

# Convert to datetime
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

# Drop invalid timestamps
df = df.dropna(subset=[time_col])

# Create daily period
df['day'] = df[time_col].dt.floor("d")

# 3. REQUIRED SENSOR COLUMNS
sensor_cols = ['temperature', 'humidity', 'light']
for c in sensor_cols:
    if c not in df.columns:
        raise KeyError(f" Missing sensor column: {c}")

# 4. DAILY AVERAGE VALUES
daily = df.groupby("day")[sensor_cols].mean().reset_index()

# 5. FIND MAX & MIN TIMES
def get_max_min_times(df, col):
    return df.loc[df[col].idxmax(), 'day'], df.loc[df[col].idxmin(), 'day']

max_temp, min_temp = get_max_min_times(daily, 'temperature')
max_hum, min_hum = get_max_min_times(daily, 'humidity')
max_light, min_light = get_max_min_times(daily, 'light')

# 6. PRINT RESULTS

print("\n DAILY MAX / MIN VALUES")
print(f" Max Temperature at: {max_temp}")
print(f" Min Temperature at: {min_temp}")
print(f" Max Humidity at: {max_hum}")
print(f" Min Humidity at: {min_hum}")
print(f" Max Light Intensity at: {max_light}")
print(f" Min Light Intensity at: {min_light}\n")

# 7. PLOTTING FUNCTION
def plot_daily(df, col, ylabel, color=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df["day"], df[col], marker='o', color=color)
    plt.title(f"{ylabel} vs Time (Daily)")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 8. PLOTS
plot_daily(daily, 'temperature', 'Temperature', color='red')
plot_daily(daily, 'humidity', 'Humidity (%)', color='green')
plot_daily(daily, 'light', 'Light Intensity', color='orange')

# WEEKLY SENSOR DATA ANALYSIS (CLEAN + VISUALIZE)

# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Clean Dataset
# Skip junk rows from merge-csv.com output
df = pd.read_csv("sensor_data_weekly.csv", skiprows=3)

# Clean column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Convert timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop any rows where timestamp could not be parsed
df = df.dropna(subset=['timestamp'])

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Check the first few rows to confirm it loaded correctly
print("\n=== CLEANED DATA SAMPLE ===")
print(df.head())

#  Correlation Heatmap

plt.figure(figsize=(8, 4))
sns.heatmap(df[['temperature', 'humidity', 'light']].corr(), annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix between Sensors')
plt.tight_layout()
plt.show()

# 1.2 Identify patterns (e.g., day–night light cycles, humidity-temperature inverse relation).
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. Load and Clean Dataset
# Skip junk rows from merge-csv.com output
df = pd.read_csv("sensor_data_weekly.csv", skiprows=3)

# Clean column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Convert timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop any rows where timestamp could not be parsed
df = df.dropna(subset=['timestamp'])

# Ensure data is sorted by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Check the first few rows to confirm it loaded correctly
print("\n=== CLEANED DATA SAMPLE ===")
print(df.head())

# 3. IDENTIFY DAY–NIGHT CYCLES
# Use light threshold (mean value)
light_threshold = df['light'].mean()
light_threshold = 200
df['day_night'] = np.where(df['light'] > light_threshold, 'Day', 'Night')

# 4. ANALYZE HUMIDITY–TEMPERATURE RELATIONSHIP
correlation = df['temperature'].corr(df['humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")

# 5. PLOT DAY–NIGHT LIGHT CYCLES
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['light'], color='gold', label='Light Intensity')
plt.fill_between(df.index, 0, df['light'],
                 where=(df['day_night'] == 'Night'),
                 color='skyblue', alpha=0.3, label='Night Period')
plt.title("Day–Night Light Cycles")
plt.xlabel("Time")
plt.ylabel("Light Intensity")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. PLOT HUMIDITY–TEMPERATURE INVERSE RELATION
fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature plot (left axis)
temp_line, = ax1.plot(df.index, df['temperature'], color='red', label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Humidity plot (right axis)
ax2 = ax1.twinx()
hum_line, = ax2.plot(df.index, df['humidity'], color='blue', label='Humidity (%)')
ax2.set_ylabel('Humidity (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends properly
lines = [temp_line, hum_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Humidity–Temperature Inverse Relationship Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

# 7. SCATTER PLOT TO VISUALIZE INVERSE RELATION
plt.figure(figsize=(12, 5))
plt.scatter(df['temperature'], df['humidity'], alpha=0.5, color='purple', label='Data Points')
plt.title("Scatter Plot: Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
# 1.3 Compute basic statistics (mean, min, max, variance per sensor).
# import liberaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
# Automatically skip junk rows until we find 'timestamp'
file_path = "sensor_data_weekly.csv"

# Step 1: Read the file once to identify where the header starts
raw_df = pd.read_csv(file_path, header=None)

# Find the row that contains the actual column headers (look for 'timestamp')
header_row = raw_df[raw_df.apply(lambda row: row.astype(str).str.contains('timestamp', case=False).any(), axis=1)].index[0]

# Step 2: Re-read the CSV using the correct header row
df = pd.read_csv(file_path, skiprows=header_row)

#  Clean up column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

#   Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop rows where timestamp is missing or invalid
df = df.dropna(subset=['timestamp'])

#  Ensure data is sorted by time
df = df.sort_values('timestamp').reset_index(drop=True)

# Print first few rows to verify
print("\n Cleaned Data Preview:")
print(df.head())

# 3. COMPUTE BASIC STATISTICS
sensor_cols = ['temperature', 'humidity', 'light']

# Calculate mean, min, max, variance for each sensor
stats_df = df[sensor_cols].agg(['mean', 'min', 'max', 'var']).T

# Rename columns for clarity
stats_df.columns = ['Mean', 'Minimum', 'Maximum', 'Variance']

# Display the computed statistics
print("=== Basic Statistics for Each Sensor ===\n")
print(stats_df)

# 4. PLOT GRAPHICAL REPRESENTATION

## (a) Bar chart of mean, min, and max
plt.figure(figsize=(10, 5))
stats_df[['Mean', 'Minimum', 'Maximum']].plot(kind='bar', figsize=(10, 5))
plt.title("Basic Sensor Statistics (Mean, Min, Max)")
plt.ylabel("Sensor Values")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (b) Variance comparison across sensors
plt.figure(figsize=(6, 4))
plt.bar(stats_df.index, stats_df['Variance'], color=['red', 'blue', 'orange'])
plt.title("Variance of Each Sensor")
plt.ylabel("Variance")
plt.xlabel("Sensor Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## (c) Boxplots to visualize distribution
plt.figure(figsize=(8, 5))
df[sensor_cols].boxplot()
plt.title("Distribution of Sensor Readings")
plt.ylabel("Sensor Values")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# (d) TIME-SERIES PLOTS WITH MAX/MIN INDICATORS

for sensor in sensor_cols:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[sensor], label=f"{sensor.capitalize()} Over Time", linewidth=1.5)

    # Identify max and min points
    max_val = df[sensor].max()
    min_val = df[sensor].min()
    max_time = df[sensor].idxmax()
    min_time = df[sensor].idxmin()

    # Plot max/min points on graph
    plt.scatter(max_time, max_val, color='red', s=60, label=f"Max: {max_val:.2f}")
    plt.scatter(min_time, min_val, color='blue', s=60, label=f"Min: {min_val:.2f}")

    # Annotate max
    plt.annotate(f"Max\n{max_val:.2f}\n{max_time}",
                 xy=(max_time, max_val),
                 xytext=(max_time, max_val + (0.05 * max_val)),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=9)

    # Annotate min
    plt.annotate(f"Min\n{min_val:.2f}\n{min_time}",
                 xy=(min_time, min_val),
                 xytext=(min_time, min_val + (0.05 * max_val)),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=9)

    # Labels
    plt.title(f"{sensor.capitalize()} Over Time with Max & Min Indicators")
    plt.xlabel("Time")
    plt.ylabel(sensor.capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
# 1.4 Anomaly Detection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#  LOAD AND CLEAN THE DATA
file_path = "sensor_data_weekly.csv"

# First read file without assuming header
raw_df = pd.read_csv(file_path, header=None)

# Detect which row contains 'timestamp' (real header)
header_row = raw_df[raw_df.apply(lambda r: r.astype(str).str.contains('timestamp', case=False).any(), axis=1)].index[0]

# Re-read the file from that header row
df = pd.read_csv(file_path, skiprows=header_row)

# Clean column names: lowercase, remove spaces
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop junk rows (like empty timestamp)
df = df.dropna(subset=['timestamp'])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# Sort chronologically
df = df.sort_values('timestamp').reset_index(drop=True)

print(" Cleaned columns:", df.columns.tolist())
print(df.head())

#  FEATURE SMOOTHING
window_size = 50  # Adjust based on sampling frequency

for col in ['temperature', 'humidity', 'light']:
    if col in df.columns:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
        df[f'{col}_anomaly'] = np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    else:
        print(f"⚠️ Warning: Column '{col}' not found in file!
        
#  ANOMALY VISUALIZATION

def plot_anomaly(sensor, color, mean_color):
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df[sensor], color=color, alpha=0.5, label=sensor.capitalize())
    plt.plot(df.index, df[f'{sensor}_rolling_mean'], color=mean_color, linewidth=2, label='Rolling Mean')
    plt.scatter(df.index[df[f'{sensor}_anomaly']],
                df[sensor][df[f'{sensor}_anomaly']],
                color='red', label='Anomalies', zorder=5)
    plt.title(f"{sensor.capitalize()} Anomaly Detection (Count: {df[f'{sensor}_anomaly'].sum()})")
    plt.xlabel("Time Index")
    plt.ylabel(sensor.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot anomalies for each sensor
for sensor, color, mean_color in [('temperature', 'gray', 'orange'),
                                  ('humidity', 'lightblue', 'red'),
                                  ('light', 'green', 'gold')]:
    if sensor in df.columns:
        plot_anomaly(sensor, color, mean_color)

# ANOMALY SUMMARY
summary = {}
for col in ['temperature', 'humidity', 'light']:
    if f'{col}_anomaly' in df.columns:
        summary[col.capitalize()] = int(df[f'{col}_anomaly'].sum())

if summary:
    print("\n=== Total Anomaly Counts ===")
    for k, v in summary.items():
        print(f"{k} anomalies: {v}")

    plt.figure(figsize=(6,4))
    plt.bar(summary.keys(), summary.values(), color=['red', 'blue', 'green'])
    plt.title("Anomaly Distribution per Sensor")
    plt.ylabel("Count of Anomalies")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD & CLEAN DATA


file_path = "sensor_data_weekly.csv"

# Attempt reading with automatic delimiter detection
data = pd.read_csv(file_path, engine="python")

# Remove units, commas, spaces from ALL columns
data = data.replace(r"[^\d\.\-eE]+", "", regex=True)

# Convert everything to numeric where possible
data = data.apply(pd.to_numeric, errors="coerce")

# Drop columns that are completely empty
data = data.dropna(axis=1, how="all")

# Drop rows that are all NaN
data = data.dropna(axis=0, how="all")

# Now select numeric columns
sensor_cols = data.select_dtypes(include=[np.number]).columns

if len(sensor_cols) == 0:
    raise ValueError("Still no numeric columns found. Check if your dataset contains valid numbers.")

print("Numeric sensor columns detected:", list(sensor_cols))


# 2. SENSOR FAULT DETECTIONS

def detect_sensor_faults(df, col, threshold=3):
    mean = df[col].mean()
    std = df[col].std()
    z_scores = (df[col] - mean) / std
    return abs(z_scores) > threshold  # boolean mask

# 3. ENVIRONMENTAL OUTLIERS

def detect_environmental_outliers(df, col, iqr_factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_factor * IQR
    upper = Q3 + iqr_factor * IQR
    return (df[col] < lower) | (df[col] > upper)

# 4. RUN DETECTIONS

results = pd.DataFrame(index=data.index)

for col in sensor_cols:
    results[f"{col}_fault"] = detect_sensor_faults(data, col)
    results[f"{col}_outlier"] = detect_environmental_outliers(data, col)

# Combined
results["any_fault"] = results.any(axis=1)

print(results.head())

# 5. PLOT

plt.figure(figsize=(12,5))
col = sensor_cols[0]  # plot first numeric sensor

plt.plot(data.index, data[col], label="Sensor Value")

fault_indices = results.index[results["any_fault"]]

plt.scatter(
    fault_indices,
    data.loc[fault_indices, col],
    color="red",
    marker="o",
    label="Detected Faults"
)

plt.legend()
plt.title("Sensor Fault & Outlier Detection")
plt.show()
# 2. FEATURE ENGINEERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  LOAD AND CLEAN THE DATA
file_path = "sensor_data_weekly.csv"

# First read file without assuming header
raw_df = pd.read_csv(file_path, header=None)

# Detect which row contains 'timestamp' (real header)
header_row = raw_df[raw_df.apply(
    lambda r: r.astype(str).str.contains('timestamp', case=False).any(),
    axis=1
)].index[0]

# Re-read the file from that header row
df = pd.read_csv(file_path, skiprows=header_row)

# Clean column names: lowercase, remove spaces
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop junk rows (like empty timestamp)
df = df.dropna(subset=['timestamp'])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# Sort chronologically
df = df.sort_values('timestamp').reset_index(drop=True)

print("Cleaned columns:", df.columns.tolist())
print(df.head())

# Do NOT re-read the file here

# Convert timestamp to datetime and set as index
df = df.set_index('timestamp')

# 2.1 Derived Features: rate of change, moving average
df['temp_rate_change'] = df['temperature'].diff()
df['hum_rate_change'] = df['humidity'].diff()

# Compute moving averages safely
df['temp_moving_avg'] = df['temperature'].rolling(window=100, min_periods=1).mean()
df['hum_moving_avg'] = df['humidity'].rolling(window=100, min_periods=1).mean()

# 2.2 Label daytime vs nighttime using light intensity
light_threshold = 100  # Adjust depending on sensor calibration
df['is_daytime'] = (df['light'] > light_threshold).astype(int)

# Add hour of day and day of week
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# 2.3 Anomaly detection for each signal
window_size = 100
for col in ['temperature', 'humidity', 'light']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
    df[f'{col}_anomaly'] = (
        np.abs(df[col] - df[f'{col}_rolling_mean']) > (2 * df[f'{col}_rolling_std'])
    )

# Combine all anomaly flags
df['anomaly_flag'] = (
    df['temperature_anomaly'] | df['humidity_anomaly'] | df['light_anomaly']
).astype(int)

# Fill NaN values
for col in [
    'temp_moving_avg', 'hum_moving_avg',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std'
]:
    df[col] = df[col].fillna(df[col].mean())

# Show NaN summary
print("\nNaN value count per feature after filling:")
print(df[['temp_moving_avg', 'hum_moving_avg',
          'temperature_rolling_mean', 'temperature_rolling_std',
          'humidity_rolling_std', 'light_rolling_mean', 'light_rolling_std']].isna().sum())

# Save
df.to_csv("processed_sensor_data.csv")

print("\nFeature Engineering complete. Saved as 'processed_sensor_data.csv'")
print(df.head())

# 2.6 FEATURE VISUALIZATION
#  Rate of Change: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_rate_change'], color='red', alpha=0.7, label='Temperature Rate of Change (°C/s)')
plt.plot(df.index, df['hum_rate_change'], color='blue', alpha=0.7, label='Humidity Rate of Change (%/s)')
plt.title("Rate of Change: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Moving Averages: Temperature vs Humidity
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temp_moving_avg'], color='orange', linewidth=2, label='Temperature Moving Average (°C)')
plt.plot(df.index, df['hum_moving_avg'], color='green', linewidth=2, label='Humidity Moving Average (%)')
plt.title("Moving Average: Temperature vs Humidity")
plt.xlabel("Time")
plt.ylabel("Moving Average Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#  Rolling Means: Temperature, Humidity, and Light
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['temperature_rolling_mean'], color='red', label='Temperature Rolling Mean (°C)')
plt.plot(df.index, df['humidity_rolling_mean'], color='blue', label='Humidity Rolling Mean (%)')
plt.plot(df.index, df['light_rolling_mean'], color='gold', label='Light Rolling Mean (Intensity)')
plt.title("Rolling Mean Comparison: Temperature, Humidity, and Light")
plt.xlabel("Time")
plt.ylabel("Rolling Mean Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

corr_features = df[['temp_rate_change', 'hum_rate_change',
                    'temp_moving_avg', 'hum_moving_avg',
                    'temperature_rolling_mean', 'humidity_rolling_mean', 'light_rolling_mean']].corr()
print("\n=== Feature Correlation Matrix ===")
print(corr_features.round(3))