# MSc IoT & Big Data – First Assignment

This repository contains my analysis of 7 days of sensor data (5-second interval) for **temperature**, **humidity**, and **light**.

The main analysis script is:

- [`iot_analysis_first_assignment.py`](iot_analysis_first_assignment.py)

All generated outputs (CSVs, plots, text report) are stored under:

- [`outputs/`](outputs/)
  - [`outputs/plots/`](outputs/plots/) – all figures
  - `iot_descriptive_stats.csv`
  - `iot_hourly_stats.csv`
  - `iot_daily_peak_hours.csv`
  - `iot_daily_summary.csv`
  - `iot_enriched_features_and_anomalies.csv`
  - `iot_story_report.txt`

---

## Task 1.1 – Time-series plots and correlation

### 1.1.1 Time-series with night shading

- Temperature over time  
  ![1.1 Temperature timeseries](outputs/plots/1.1_temperature_timeseries.png)

- Humidity over time  
  ![1.1 Humidity timeseries](outputs/plots/1.1_humidity_timeseries.png)

- Light intensity over time  
  ![1.1 Light timeseries](outputs/plots/1.1_light_timeseries.png)

### 1.1.2 Temperature & Humidity relationship

- Simple scatter plot  
  ![1.1 Temperature vs Humidity scatter](outputs/plots/1.1_temperature_vs_humidity_scatter.png)

- Scatter + interpretation proof  
  ![1.1 Temperature vs Humidity scatter proof](outputs/plots/1.1_temperature_vs_humidity_scatter_proof.png)

- Full-week dual time-series (temperature & humidity)  
  ![1.1 Temperature & Humidity timeseries proof](outputs/plots/1.1_temperature_humidity_timeseries_proof.png)

### 1.1.3 Correlation heatmap

- Correlation between temperature, humidity, and light  
  ![1.1 Correlation heatmap](outputs/plots/1.1_correlation_heatmap.png)

---

## Task 1.2 – Day vs Night comparisons

Day/night labels are based on time (06:00–18:00 = day, otherwise night).

- Temperature – Day vs Night  
  ![1.2 Temperature Day vs Night](outputs/plots/1.2_temperature_day_night_box.png)

- Humidity – Day vs Night  
  ![1.2 Humidity Day vs Night](outputs/plots/1.2_humidity_day_night_box.png)

---

## Task 1.3 – Descriptive and hourly statistics

Descriptive statistics of all sensors are saved in:

- [`outputs/iot_descriptive_stats.csv`](outputs/iot_descriptive_stats.csv)

Hourly mean/min/max statistics are saved in:

- [`outputs/iot_hourly_stats.csv`](outputs/iot_hourly_stats.csv)

### 1.3.1 Per-sensor hourly mean/min/max

- Temperature – hourly mean / min / max  
  ![1.3 Temperature hourly mean/min/max](outputs/plots/1.3_temperature_hourly_mean_min_max.png)

- Humidity – hourly mean / min / max  
  ![1.3 Humidity hourly mean/min/max](outputs/plots/1.3_humidity_hourly_mean_min_max.png)

- Light – hourly mean / min / max  
  ![1.3 Light hourly mean/min/max](outputs/plots/1.3_light_hourly_mean_min_max.png)

### 1.3.2 Combined hourly view

- Temperature, Humidity, Light – hourly mean/min/max in one figure  
  ![1.3 All sensors hourly mean/min/max](outputs/plots/1.3_all_sensors_hourly_mean_min_max.png)

---

## Task 1.4 – Daily peak hours (hottest / most humid / brightest hour)

Daily peak hours (based on hourly means) are stored in:

- [`outputs/iot_daily_peak_hours.csv`](outputs/iot_daily_peak_hours.csv)

### 1.4.1 Summary bar plots (per day)

- Hottest hour of each day (temperature)  
  ![1.4 Daily hottest hours – temperature](outputs/plots/1.4_daily_hottest_hours_temperature.png)

- Most humid hour of each day (humidity)  
  ![1.4 Daily most humid hours](outputs/plots/1.4_daily_most_humid_hours.png)

- Brightest hour of each day (light)  
  ![1.4 Daily brightest hours](outputs/plots/1.4_daily_brightest_hours.png)

### 1.4.2 Hour-of-day vs true hourly max, per variable and per day

#### Temperature – hourly maxima by day

- 2025-03-01  
  ![1.4 Temperature hourly max 2025-03-01](outputs/plots/1.4_temperature_hourly_max_2025-03-01.png)
- 2025-03-02  
  ![1.4 Temperature hourly max 2025-03-02](outputs/plots/1.4_temperature_hourly_max_2025-03-02.png)
- 2025-03-03  
  ![1.4 Temperature hourly max 2025-03-03](outputs/plots/1.4_temperature_hourly_max_2025-03-03.png)
- 2025-03-04  
  ![1.4 Temperature hourly max 2025-03-04](outputs/plots/1.4_temperature_hourly_max_2025-03-04.png)
- 2025-03-05  
  ![1.4 Temperature hourly max 2025-03-05](outputs/plots/1.4_temperature_hourly_max_2025-03-05.png)
- 2025-03-06  
  ![1.4 Temperature hourly max 2025-03-06](outputs/plots/1.4_temperature_hourly_max_2025-03-06.png)
- 2025-03-07  
  ![1.4 Temperature hourly max 2025-03-07](outputs/plots/1.4_temperature_hourly_max_2025-03-07.png)

- All days combined  
  ![1.4 Temperature hourly max – all days](outputs/plots/1.4_temperature_hourly_max_all_days.png)

#### Humidity – hourly maxima by day

- 2025-03-01  
  ![1.4 Humidity hourly max 2025-03-01](outputs/plots/1.4_humidity_hourly_max_2025-03-01.png)
- 2025-03-02  
  ![1.4 Humidity hourly max 2025-03-02](outputs/plots/1.4_humidity_hourly_max_2025-03-02.png)
- 2025-03-03  
  ![1.4 Humidity hourly max 2025-03-03](outputs/plots/1.4_humidity_hourly_max_2025-03-03.png)
- 2025-03-04  
  ![1.4 Humidity hourly max 2025-03-04](outputs/plots/1.4_humidity_hourly_max_2025-03-04.png)
- 2025-03-05  
  ![1.4 Humidity hourly max 2025-03-05](outputs/plots/1.4_humidity_hourly_max_2025-03-05.png)
- 2025-03-06  
  ![1.4 Humidity hourly max 2025-03-06](outputs/plots/1.4_humidity_hourly_max_2025-03-06.png)
- 2025-03-07  
  ![1.4 Humidity hourly max 2025-03-07](outputs/plots/1.4_humidity_hourly_max_2025-03-07.png)

- All days combined  
  ![1.4 Humidity hourly max – all days](outputs/plots/1.4_humidity_hourly_max_all_days.png)

#### Light – hourly maxima by day

- 2025-03-01  
  ![1.4 Light hourly max 2025-03-01](outputs/plots/1.4_light_hourly_max_2025-03-01.png)
- 2025-03-02  
  ![1.4 Light hourly max 2025-03-02](outputs/plots/1.4_light_hourly_max_2025-03-02.png)
- 2025-03-03  
  ![1.4 Light hourly max 2025-03-03](outputs/plots/1.4_light_hourly_max_2025-03-03.png)
- 2025-03-04  
  ![1.4 Light hourly max 2025-03-04](outputs/plots/1.4_light_hourly_max_2025-03-04.png)
- 2025-03-05  
  ![1.4 Light hourly max 2025-03-05](outputs/plots/1.4_light_hourly_max_2025-03-05.png)
- 2025-03-06  
  ![1.4 Light hourly max 2025-03-06](outputs/plots/1.4_light_hourly_max_2025-03-06.png)
- 2025-03-07  
  ![1.4 Light hourly max 2025-03-07](outputs/plots/1.4_light_hourly_max_2025-03-07.png)

- All days combined  
  ![1.4 Light hourly max – all days](outputs/plots/1.4_light_hourly_max_all_days.png)

---

## Task 1.5 – Daily mean temperature, humidity, and light

Daily mean values and the hottest/most humid/brightest day are stored in:

- [`outputs/iot_daily_summary.csv`](outputs/iot_daily_summary.csv)

Plots of daily means:

- Daily mean temperature  
  ![1.5 Daily mean temperature](outputs/plots/1.5_daily_mean_temperature.png)

- Daily mean humidity  
  ![1.5 Daily mean humidity](outputs/plots/1.5_daily_mean_humidity.png)

- Daily mean light intensity  
  ![1.5 Daily mean light](outputs/plots/1.5_daily_mean_light.png)

---

## Task 2 – Feature engineering & anomaly detection

### 2.1 Moving average (10-min) and rate of change (ROC)

Weekly overview:

- Temperature – MA & time-series (whole week)  
  ![2.1 Temperature MA (week)](outputs/plots/2.1_temperature_ma.png)

- Temperature – ROC (whole week)  
  ![2.1 Temperature ROC (week)](outputs/plots/2.1_temperature_roc.png)

- Humidity – MA (whole week)  
  ![2.1 Humidity MA (week)](outputs/plots/2.1_humidity_ma.png)

- Humidity – ROC (whole week)  
  ![2.1 Humidity ROC (week)](outputs/plots/2.1_humidity_roc.png)

Per-day MA plots (temperature):

- 2025-03-01  
  ![2.1 Temperature MA 2025-03-01](outputs/plots/2.1_temperature_ma_2025-03-01.png)
- 2025-03-02  
  ![2.1 Temperature MA 2025-03-02](outputs/plots/2.1_temperature_ma_2025-03-02.png)
- 2025-03-03  
  ![2.1 Temperature MA 2025-03-03](outputs/plots/2.1_temperature_ma_2025-03-03.png)
- 2025-03-04  
  ![2.1 Temperature MA 2025-03-04](outputs/plots/2.1_temperature_ma_2025-03-04.png)
- 2025-03-05  
  ![2.1 Temperature MA 2025-03-05](outputs/plots/2.1_temperature_ma_2025-03-05.png)
- 2025-03-06  
  ![2.1 Temperature MA 2025-03-06](outputs/plots/2.1_temperature_ma_2025-03-06.png)
- 2025-03-07  
  ![2.1 Temperature MA 2025-03-07](outputs/plots/2.1_temperature_ma_2025-03-07.png)

Per-day ROC plots (temperature):

- 2025-03-01  
  ![2.1 Temperature ROC 2025-03-01](outputs/plots/2.1_temperature_roc_2025-03-01.png)
- 2025-03-02  
  ![2.1 Temperature ROC 2025-03-02](outputs/plots/2.1_temperature_roc_2025-03-02.png)
- 2025-03-03  
  ![2.1 Temperature ROC 2025-03-03](outputs/plots/2.1_temperature_roc_2025-03-03.png)
- 2025-03-04  
  ![2.1 Temperature ROC 2025-03-04](outputs/plots/2.1_temperature_roc_2025-03-04.png)
- 2025-03-05  
  ![2.1 Temperature ROC 2025-03-05](outputs/plots/2.1_temperature_roc_2025-03-05.png)
- 2025-03-06  
  ![2.1 Temperature ROC 2025-03-06](outputs/plots/2.1_temperature_roc_2025-03-06.png)
- 2025-03-07  
  ![2.1 Temperature ROC 2025-03-07](outputs/plots/2.1_temperature_roc_2025-03-07.png)

Per-day MA plots (humidity):

- 2025-03-01  
  ![2.1 Humidity MA 2025-03-01](outputs/plots/2.1_humidity_ma_2025-03-01.png)
- 2025-03-03  
  ![2.1 Humidity MA 2025-03-03](outputs/plots/2.1_humidity_ma_2025-03-03.png)
- 2025-03-04  
  ![2.1 Humidity MA 2025-03-04](outputs/plots/2.1_humidity_ma_2025-03-04.png)
- 2025-03-05  
  ![2.1 Humidity MA 2025-03-05](outputs/plots/2.1_humidity_ma_2025-03-05.png)
- 2025-03-06  
  ![2.1 Humidity MA 2025-03-06](outputs/plots/2.1_humidity_ma_2025-03-06.png)
- 2025-03-07  
  ![2.1 Humidity MA 2025-03-07](outputs/plots/2.1_humidity_ma_2025-03-07.png)

Per-day ROC plots (humidity):

- 2025-03-01  
  ![2.1 Humidity ROC 2025-03-01](outputs/plots/2.1_humidity_roc_2025-03-01.png)
- 2025-03-02  
  ![2.1 Humidity ROC 2025-03-02](outputs/plots/2.1_humidity_roc_2025-03-02.png)
- 2025-03-03  
  ![2.1 Humidity ROC 2025-03-03](outputs/plots/2.1_humidity_roc_2025-03-03.png)
- 2025-03-04  
  ![2.1 Humidity ROC 2025-03-04](outputs/plots/2.1_humidity_roc_2025-03-04.png)
- 2025-03-05  
  ![2.1 Humidity ROC 2025-03-05](outputs/plots/2.1_humidity_roc_2025-03-05.png)
- 2025-03-06  
  ![2.1 Humidity ROC 2025-03-06](outputs/plots/2.1_humidity_roc_2025-03-06.png)
- 2025-03-07  
  ![2.1 Humidity ROC 2025-03-07](outputs/plots/2.1_humidity_roc_2025-03-07.png)

### 2.2 Anomaly detection (±3σ, 10-minute rolling window)

Anomalies are detected using a 3σ rule on a 10-minute rolling mean and standard deviation.

- Temperature anomalies  
  ![2.2 Temperature anomalies](outputs/plots/2.2_temperature_anomalies.png)

- Humidity anomalies  
  ![2.2 Humidity anomalies](outputs/plots/2.2_humidity_anomalies.png)

- Light anomalies  
  ![2.2 Light anomalies](outputs/plots/2.2_light_anomalies.png)

---

## Story-like interpretation

A narrative summary of the dataset (environment behaviour, day–night patterns, correlations, and anomalies) is saved as:

- [`outputs/iot_story_report.txt`](outputs/iot_story_report.txt)
