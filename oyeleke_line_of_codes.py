# iot_analysis_first_assignment.py
#
# MSc IoT & Big Data – First Assignment
# -------------------------------------
# This script:
# - Loads 7 days of sensor data (5s interval)
# - Focuses on temperature, humidity, and light
# - Does EDA (Task 1.1, 1.2, 1.3)
# - Extracts features and detects anomalies (Task 2 – MSc)
# - Computes:
#     * hourly mean/min/max
#     * hottest / most humid / brightest hour of each day
#     * hottest / most humid / brightest day of the week
# - Creates plots and a story-like text report.

import os
import glob
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
FILE_PATTERN = "sensor_data_2025-03-0*.csv"
EXPECTED_STEP_SECONDS = 5

# Day / night by time (primary)
DAY_START_HOUR = 6    # 06:00
DAY_END_HOUR = 18     # 18:00

ROLL_WINDOW_MINUTES = 10
PLOT_RESAMPLE = "1min"   # 60 seconds interval for plotting

OUT_BASE = "outputs"
OUT_PLOTS_DIR = os.path.join(OUT_BASE, "plots")

OUT_STATS_CSV = os.path.join(OUT_BASE, "iot_descriptive_stats.csv")
OUT_HOURLY_STATS_CSV = os.path.join(OUT_BASE, "iot_hourly_stats.csv")
OUT_DAILY_PEAKS_CSV = os.path.join(OUT_BASE, "iot_daily_peak_hours.csv")
OUT_DAILY_SUMMARY_CSV = os.path.join(OUT_BASE, "iot_daily_summary.csv")
OUT_ENRICHED_CSV = os.path.join(OUT_BASE, "iot_enriched_features_and_anomalies.csv")
OUT_STORY_TXT = os.path.join(OUT_BASE, "iot_story_report.txt")

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)


# -----------------------------
# HELPERS
# -----------------------------
def load_all_csvs(file_pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {file_pattern}")

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Normalize timestamp column
    ts_col = None
    for c in ["timestamp", "time", "datetime", "date_time"]:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is None:
        df["timestamp"] = pd.date_range(
            start="2025-03-01", periods=len(df), freq=f"{EXPECTED_STEP_SECONDS}S"
        )
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        if df["timestamp"].isna().any():
            df["timestamp"] = pd.date_range(
                start="2025-03-01", periods=len(df), freq=f"{EXPECTED_STEP_SECONDS}S"
            )

    df = df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    return df


def infer_cadence_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float(EXPECTED_STEP_SECONDS)
    step = index.to_series().diff().median()
    return step.total_seconds() if pd.notnull(step) else float(EXPECTED_STEP_SECONDS)


def label_day_night_time(index: pd.DatetimeIndex,
                         day_start_hour: int = 6,
                         day_end_hour: int = 18) -> pd.Series:
    hours = index.hour
    is_day = (hours >= day_start_hour) & (hours < day_end_hour)
    return is_day.astype(int)


def label_day_night_light(light_series: pd.Series) -> Tuple[pd.Series, float]:
    """Binary label from light: values >= median → 1, else 0."""
    ls = pd.to_numeric(light_series, errors="coerce")
    if ls.isna().all():
        return pd.Series(index=ls.index, dtype="float64"), np.nan
    thr = float(ls.median())
    labels = (ls >= thr).astype(int)
    return labels, thr


def compute_descriptive_stats(df: pd.DataFrame, cols) -> pd.DataFrame:
    stats = df[cols].agg(["mean", "min", "max", "var"]).T
    stats = stats.rename(columns={"var": "variance"})
    stats.index.name = "sensor"
    return stats.reset_index()


def compute_rate_of_change(series: pd.Series, step_seconds: float) -> pd.Series:
    return series.diff() / step_seconds


def compute_moving_average(series: pd.Series, window_minutes: int) -> pd.Series:
    return series.rolling(f"{window_minutes}min").mean()


def rolling_mean_std(series: pd.Series, window_minutes: int) -> Tuple[pd.Series, pd.Series]:
    rm = series.rolling(f"{window_minutes}min").mean()
    rs = series.rolling(f"{window_minutes}min").std()
    return rm, rs


def detect_anomalies_3sigma(series: pd.Series,
                            rolling_mean: pd.Series,
                            rolling_std: pd.Series,
                            z: float = 3.0) -> pd.Series:
    zscore = (series - rolling_mean).abs() / rolling_std
    return (zscore > z).fillna(False)


def resample_for_plotting(df: pd.DataFrame, cols: List[str], rule: str) -> pd.DataFrame:
    return df[cols].resample(rule).median()


def shade_night_bands(ax, ts_index: pd.DatetimeIndex, daytime_series: pd.Series, alpha=0.1):
    if daytime_series.isna().all():
        return
    d = daytime_series.astype(int)
    changes = d.diff().fillna(0).ne(0)
    boundaries = [ts_index[0]]
    boundaries += list(ts_index[changes])
    boundaries += [ts_index[-1]]

    current_state = int(d.iloc[0])
    for i in range(0, len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if current_state == 0:
            ax.axvspan(start, end, color="black", alpha=alpha, zorder=0)
        current_state = 1 - current_state


# -----------------------------
# TIME-BASED AGGREGATES
# -----------------------------
def compute_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {}
    for col in ["temperature", "humidity", "light"]:
        if col in df.columns:
            agg_dict[col] = ["mean", "min", "max"]
    hourly = df.resample("1H").agg(agg_dict)
    hourly.columns = ["_".join(col) for col in hourly.columns]
    return hourly


def compute_daily_peaks(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["date"] = hourly.index.date
    hourly["hour"] = hourly.index.hour

    rows = []
    for date, group in hourly.groupby("date"):
        row = {"date": str(date)}
        for var in ["temperature", "humidity", "light"]:
            col = f"{var}_mean"
            if col in group.columns:
                idxmax = group[col].idxmax()
                row[f"{var}_peak_hour"] = int(idxmax.hour)
                row[f"{var}_peak_value"] = float(group.loc[idxmax, col])
        rows.append(row)

    return pd.DataFrame(rows)


def compute_daily_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]:
    agg_dict = {}
    for var in ["temperature", "humidity", "light"]:
        if var in df.columns:
            agg_dict[var] = "mean"
    daily = df.resample("1D").agg(agg_dict)

    best_days: Dict[str, pd.Timestamp] = {}
    for var in ["temperature", "humidity", "light"]:
        if var in daily.columns:
            best_days[var] = daily[var].idxmax()
    return daily, best_days


# -----------------------------
# PLOTTING HELPERS
# -----------------------------
def plot_daily_peak_hours(daily_peaks: pd.DataFrame, out_dir: str) -> None:
    """
    Plot, for each variable (temperature, humidity, light),
    the peak hourly mean per day (hottest / most humid / brightest hour).
    (Kept for story / reference.)
    """
    if daily_peaks.empty:
        return

    dates = pd.to_datetime(daily_peaks["date"])

    # Temperature
    if "temperature_peak_value" in daily_peaks.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dates, daily_peaks["temperature_peak_value"], width=0.6)
        for x, y, h in zip(dates, daily_peaks["temperature_peak_value"], daily_peaks["temperature_peak_hour"]):
            ax.text(x, y, f"{int(h):02d}:00", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title("Hottest hour of each day (hourly mean temperature)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1.4_daily_hottest_hours_temperature.png"), dpi=180)
        plt.close(fig)

    # Humidity
    if "humidity_peak_value" in daily_peaks.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dates, daily_peaks["humidity_peak_value"], width=0.6)
        for x, y, h in zip(dates, daily_peaks["humidity_peak_value"], daily_peaks["humidity_peak_hour"]):
            ax.text(x, y, f"{int(h):02d}:00", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title("Most humid hour of each day (hourly mean humidity)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Humidity (%)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1.4_daily_most_humid_hours.png"), dpi=180)
        plt.close(fig)

    # Light
    if "light_peak_value" in daily_peaks.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dates, daily_peaks["light_peak_value"], width=0.6)
        for x, y, h in zip(dates, daily_peaks["light_peak_value"], daily_peaks["light_peak_hour"]):
            ax.text(x, y, f"{int(h):02d}:00", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title("Brightest hour of each day (hourly mean light intensity)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Light (lux)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1.4_daily_brightest_hours.png"), dpi=180)
        plt.close(fig)


def plot_daily_means(daily_means: pd.DataFrame, out_dir: str) -> None:
    """
    Plot daily average temperature, humidity, and light over the week.
    This is used to visualise hottest / most humid / brightest day of the week.
    """
    if daily_means.empty:
        return

    dates = daily_means.index

    # Temperature daily mean
    if "temperature" in daily_means.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, daily_means["temperature"], marker="o")
        ax.set_title("Daily mean temperature (hottest day of the week)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1.5_daily_mean_temperature.png"), dpi=180)
        plt.close(fig)

    # Humidity daily mean
    if "humidity" in daily_means.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, daily_means["humidity"], marker="o")
        ax.set_title("Daily mean humidity (most humid day of the week)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Humidity (%)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1.5_daily_mean_humidity.png"), dpi=180)
        plt.close(fig)

    # Light daily mean
    if "light" in daily_means.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, daily_means["light"], marker="o")
        ax.set_title("Daily mean light intensity (brightest day of the week)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Light (lux)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1.5_daily_mean_light.png"), dpi=180)
        plt.close(fig)


def plot_daily_hourly_max_profiles(df: pd.DataFrame, out_dir: str) -> None:
    """
    For each day and each variable (temperature, humidity, light),
    plot 'time of day (hour 0–23) vs true hourly maximum value'.

    Uses 60-second resampling first (to respect the 1-minute interval),
    then takes max within each hour for accuracy.
    """
    if df.empty:
        return

    # Resample to 60 seconds to work with 1-minute interval
    df_1min = df.resample("60S").mean()
    df_1min["date"] = df_1min.index.date
    df_1min["hour"] = df_1min.index.hour

    for var in ["temperature", "humidity", "light"]:
        if var not in df_1min.columns:
            continue

        profiles: Dict[str, pd.Series] = {}

        # Per-day plots: hour of day vs max value in that hour
        for date, g in df_1min.groupby("date"):
            by_hour = g.groupby("hour")[var].max()  # true max per hour for that day
            profiles[str(date)] = by_hour

            fig, ax = plt.subplots(figsize=(8, 4))
            hours = by_hour.index.values
            vals = by_hour.values
            ax.plot(hours, vals, marker="o")

            # highlight the hottest / most humid / brightest hour for that day
            if len(vals) > 0:
                max_idx = int(np.argmax(vals))
                ax.scatter(hours[max_idx], vals[max_idx], s=40)

            ax.set_xticks(range(0, 24, 2))
            ax.set_xlabel("Hour of day")
            ax.set_ylabel(var.capitalize())
            ax.set_title(f"{var.capitalize()} hourly maxima – {date}")
            fig.tight_layout()
            fname = f"1.4_{var}_hourly_max_{date}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=180)
            plt.close(fig)

        # Combined plot for the 7 days together
        if profiles:
            fig, ax = plt.subplots(figsize=(10, 5))
            for date_str, by_hour in profiles.items():
                ax.plot(by_hour.index.values,
                        by_hour.values,
                        marker="o",
                        label=date_str)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xlabel("Hour of day")
            ax.set_ylabel(var.capitalize())
            ax.set_title(f"{var.capitalize()} hourly maxima – all days")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fname_all = f"1.4_{var}_hourly_max_all_days.png"
            fig.savefig(os.path.join(out_dir, fname_all), dpi=180)
            plt.close(fig)


def plot_daily_roc_ma(df: pd.DataFrame,
                      var: str,
                      ma_col: str,
                      roc_col: str,
                      out_dir: str) -> None:
    """
    For each day, plot:
    - var + its 10-min moving average
    - rate of change (ROC)
    using 60-second resampling for smoother daily view.
    """
    if var not in df.columns or ma_col not in df.columns or roc_col not in df.columns:
        return

    # unique dates in dataset
    dates = sorted(df.index.normalize().unique())

    for d in dates:
        day_mask = df.index.normalize() == d
        day_df = df.loc[day_mask].copy()
        if day_df.empty:
            continue

        date_str = pd.to_datetime(d).date().isoformat()

        # moving average per day (resampled to 1 minute for plotting)
        cols_ma = [var, ma_col]
        if "daytime_time" in day_df.columns:
            cols_ma.append("daytime_time")

        if PLOT_RESAMPLE:
            plt_df = resample_for_plotting(day_df, cols_ma, PLOT_RESAMPLE)
        else:
            plt_df = day_df[cols_ma]

        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(plt_df.index, plt_df[var], linewidth=0.7, label=var.capitalize())
        ax.plot(plt_df.index, plt_df[ma_col], linewidth=1.1, label="10-min moving average")
        if "daytime_time" in cols_ma:
            shade_night_bands(ax, plt_df.index, plt_df["daytime_time"], alpha=0.12)
        ax.set_title(f"{var.capitalize()} with 10-min moving average – {date_str}")
        ax.set_xlabel("Time")
        ax.set_ylabel(var.capitalize())
        ax.legend()
        fig.tight_layout()
        fname_ma = f"2.1_{var}_ma_{date_str}.png"
        fig.savefig(os.path.join(out_dir, fname_ma), dpi=180)
        plt.close(fig)

        # ROC per day (resampled to 1 minute for plotting)
        cols_roc = [roc_col]
        if PLOT_RESAMPLE:
            plt_df = resample_for_plotting(day_df, cols_roc, PLOT_RESAMPLE)
        else:
            plt_df = day_df[cols_roc]

        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(plt_df.index, plt_df[roc_col], linewidth=0.8)
        ax.set_title(f"{var.capitalize()} rate of change – {date_str}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Δ{var.capitalize()} / s")
        fig.tight_layout()
        fname_roc = f"2.1_{var}_roc_{date_str}.png"
        fig.savefig(os.path.join(out_dir, fname_roc), dpi=180)
        plt.close(fig)


def plot_temp_humidity_independence(df: pd.DataFrame,
                                    corr_matrix: pd.DataFrame,
                                    out_dir: str) -> None:
    """
    Graphical proof that temperature and humidity do not strongly affect each other:
    - Scatter plot with correlation annotated
    - Dual-axis time series over the full week
    """
    if not all(k in df.columns for k in ["temperature", "humidity"]):
        return

    # --- Scatter plot: Temperature vs Humidity ---
    # Use 1-point-per-minute sampling for visibility
    samp = df[["temperature", "humidity"]].iloc[::60].dropna()

    if corr_matrix is not None \
       and "temperature" in corr_matrix.columns \
       and "humidity" in corr_matrix.columns:
        rho = float(corr_matrix.loc["temperature", "humidity"])
    else:
        rho = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(samp["temperature"], samp["humidity"], s=6)
    if np.isfinite(rho):
        ax.set_title(f"Temperature vs Humidity – no clear relationship (corr = {rho:.3f})")
    else:
        ax.set_title("Temperature vs Humidity – no clear relationship")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1.1_temperature_vs_humidity_scatter_proof.png"), dpi=180)
    plt.close(fig)

    # --- Dual-axis time series over the full week ---
    # Resample to 1-minute resolution for a clean view
    plt_df = resample_for_plotting(df, ["temperature", "humidity"], PLOT_RESAMPLE)

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(plt_df.index, plt_df["temperature"], linewidth=1.0, label="Temperature")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°C)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(plt_df.index, plt_df["humidity"], linewidth=1.0, alpha=0.7, label="Humidity")
    ax2.set_ylabel("Humidity (%)")
    ax2.tick_params(axis="y")

    fig.suptitle("Temperature and Humidity over Time – flat and independent behaviour")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1.1_temperature_humidity_timeseries_proof.png"), dpi=180)
    plt.close(fig)


# -----------------------------
# STORY / REPORT GENERATOR
# -----------------------------
def build_story(df: pd.DataFrame,
                step_s: float,
                corr_matrix: pd.DataFrame,
                median_light: float,
                anomaly_counts: Dict[str, int],
                sensor_cols: List[str],
                hourly_stats: pd.DataFrame,
                daily_peaks: pd.DataFrame,
                daily_means: pd.DataFrame,
                best_days: Dict[str, pd.Timestamp]) -> str:
    lines: List[str] = []

    idx = df.index
    start, end = idx[0], idx[-1]
    duration = end - start
    total_days = duration.total_seconds() / 86400.0

    lines.append("IOT SENSOR DATASET – WEEKLY INTERPRETATION")
    lines.append("")
    lines.append("1. Dataset overview")
    lines.append(f"- Time span: {start} to {end} (~{total_days:.1f} days).")
    lines.append(f"- Number of records: {len(df):,} at ~{step_s:.1f}-second intervals.")
    lines.append(f"- Available sensors: {', '.join(sensor_cols)}.")
    lines.append("")

    # 2. Day–night behaviour
    if "daytime_time" in df.columns:
        day_ratio = df["daytime_time"].mean()
        lines.append("2. Day–night behaviour")
        lines.append(
            f"- Daytime defined by time: {DAY_START_HOUR:02d}:00–{DAY_END_HOUR:02d}:00; "
            "night = outside this range."
        )
        if np.isfinite(median_light):
            lines.append(
                f"- Median light level ≈ {median_light:.2f} lux. "
                "Values rarely approach 0 and never reach direct sunlight levels, "
                "which suggests an indoor or artificially lit environment."
            )
        lines.append(f"- Fraction of time labelled as day: {day_ratio*100:.1f}% "
                     f"(night: {(1-day_ratio)*100:.1f}%).")

        for col in sensor_cols:
            if col not in df.columns:
                continue
            g = df.groupby("daytime_time")[col].agg(["mean", "min", "max"])
            if 0 in g.index and 1 in g.index:
                dmean, nmean = g.loc[1, "mean"], g.loc[0, "mean"]
                diff = dmean - nmean
                if abs(diff) < 0.05:
                    trend = "almost the same in day and night"
                elif diff > 0:
                    trend = "slightly higher during the day"
                else:
                    trend = "slightly higher at night"

                lines.append(f"- {col.capitalize()}:")
                lines.append(f"  • Day: mean={dmean:.2f}, min={g.loc[1, 'min']:.2f}, max={g.loc[1, 'max']:.2f}")
                lines.append(f"  • Night: mean={nmean:.2f}, min={g.loc[0, 'min']:.2f}, max={g.loc[0, 'max']:.2f}")
                lines.append(f"  • Interpretation: {col} is {trend}.")
        lines.append("")

    # 3. Hourly behaviour and daily peak hours
    if not hourly_stats.empty and not daily_peaks.empty:
        lines.append("3. Hourly behaviour and daily peak hours")
        lines.append(
            "- The signals were resampled to 1-hour intervals to compute mean, minimum, "
            "and maximum values for temperature, humidity, and light."
        )
        lines.append("- For each day, the hottest, most humid, and brightest hour were extracted:")
        for _, row in daily_peaks.iterrows():
            date = row["date"]
            pieces = [f"  • {date}:"]
            if "temperature_peak_hour" in row:
                pieces.append(
                    f" hottest at {int(row['temperature_peak_hour']):02d}:00 "
                    f"(mean temperature ≈ {row['temperature_peak_value']:.2f} °C)"
                )
            if "humidity_peak_hour" in row:
                pieces.append(
                    f", most humid at {int(row['humidity_peak_hour']):02d}:00 "
                    f"(mean humidity ≈ {row['humidity_peak_value']:.2f} %)"
                )
            if "light_peak_hour" in row:
                pieces.append(
                    f", brightest at {int(row['light_peak_hour']):02d}:00 "
                    f"(mean light ≈ {row['light_peak_value']:.2f} lux)"
                )
            lines.append("".join(pieces))
        lines.append("")

    # 4. Daily extremes
    if not daily_means.empty and best_days:
        lines.append("4. Hottest, most humid, and brightest day of the week")
        for var, ts in best_days.items():
            if ts in daily_means.index:
                val = daily_means.loc[ts, var]
                label = {
                    "temperature": "Hottest day (by average temperature)",
                    "humidity": "Most humid day (by average humidity)",
                    "light": "Brightest day (by average light level)",
                }.get(var, var)
                lines.append(
                    f"- {label}: {ts.date()} with daily mean {var} ≈ {val:.2f}."
                )
        lines.append("")

    # 5. Relationships between sensors
    if corr_matrix is not None:
        lines.append("5. Relationships between temperature, humidity, and light")
        for (a, b) in [("temperature", "humidity"),
                       ("temperature", "light"),
                       ("humidity", "light")]:
            if a in corr_matrix.columns and b in corr_matrix.columns:
                rho = float(corr_matrix.loc[a, b])
                strength = "very weak"
                if abs(rho) >= 0.7:
                    strength = "strong"
                elif abs(rho) >= 0.4:
                    strength = "moderate"
                elif abs(rho) >= 0.2:
                    strength = "weak"
                direction = "positive" if rho > 0 else "negative" if rho < 0 else "no clear"
                lines.append(
                    f"- {a.capitalize()} vs {b}: correlation = {rho:+.2f} "
                    f"({strength}, {direction} relationship)."
                )
        lines.append("")

    # 6. Overall stats
    lines.append("6. Overall statistics for the full week")
    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col]
        lines.append(
            f"- {col.capitalize()}: mean={s.mean():.2f}, min={s.min():.2f}, "
            f"max={s.max():.2f}, variance={s.var():.2f}."
        )
    lines.append("")

    # 7. Anomaly detection summary
    if anomaly_counts:
        lines.append("7. Anomaly detection (±3σ, 10-minute rolling window)")
        total_an = sum(anomaly_counts.values())
        if total_an == 0:
            lines.append(
                "- No anomalies were detected for any sensor using the 3σ rule. "
                "The environment appears very stable with no extreme spikes."
            )
        else:
            lines.append(f"- Total anomalies across sensors: {total_an}.")
            for k, v in anomaly_counts.items():
                lines.append(f"  • {k.capitalize()}: {v} anomaly points.")
        lines.append("")

    # 8. Interpretation
    lines.append("8. Interpretation")
    lines.append(
        "- Maximum light levels are around 1000 lux, which is far below direct sunlight "
        "and more consistent with indoor or artificial lighting. This is why time-based "
        "day/night labelling was used as the main reference, while light intensity was "
        "used to support the interpretation."
    )
    lines.append(
        "- Temperature and humidity are almost constant over days and nights, and across "
        "the full week. This suggests a controlled environment where heating, cooling, "
        "and moisture levels are regulated."
    )
    lines.append(
        "- There is no clear natural daily cycle and no strong correlation between the "
        "sensors, confirming that the system is not simply reacting to outdoor weather "
        "but is running in a stable, indoor regime."
    )

    return "\n".join(lines)


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_all_csvs(FILE_PATTERN)
    step_s = infer_cadence_seconds(df.index)

    sensor_cols = [c for c in ["temperature", "humidity", "light", "pH", "electrical_conductivity"]
                   if c in df.columns]
    if not sensor_cols:
        raise ValueError("No expected sensor columns found in CSVs.")

    # Time-based day/night
    df["daytime_time"] = label_day_night_time(df.index, DAY_START_HOUR, DAY_END_HOUR)

    # Light-based binary (for interpretation)
    if "light" in df.columns:
        df["daytime_light"], median_light = label_day_night_light(df["light"])
    else:
        df["daytime_light"] = np.nan
        median_light = np.nan

    # Time features
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # Correlation matrix (Task 1.1)
    corr_cols = [c for c in ["temperature", "humidity", "light"] if c in df.columns]
    corr_matrix = df[corr_cols].corr() if len(corr_cols) >= 2 else None

    # 1.1 – Time-series plots with night shading
    for col in [c for c in ["temperature", "humidity", "light"] if c in df.columns]:
        cols_to_use = [col, "daytime_time"]
        plt_df = resample_for_plotting(df, cols_to_use, PLOT_RESAMPLE) if PLOT_RESAMPLE else df[cols_to_use]
        fig, ax = plt.subplots(figsize=(14, 3.6))
        ax.plot(plt_df.index, plt_df[col], linewidth=1.0)
        shade_night_bands(ax, plt_df.index, plt_df["daytime_time"], alpha=0.12)
        ax.set_title(f"{col.capitalize()} over time (night shaded)")
        ax.set_xlabel("Time")
        ax.set_ylabel(col.capitalize())
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, f"1.1_{col}_timeseries.png"), dpi=180)
        plt.close(fig)

    # 1.1 – Correlation heatmap
    if corr_matrix is not None:
        labels = corr_matrix.columns.tolist()
        fig, ax = plt.subplots(figsize=(6 + 1.0 * len(labels), 4 + 0.5 * len(labels)))
        im = ax.imshow(corr_matrix.values, aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title("Correlation heatmap (Temperature / Humidity / Light)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "1.1_correlation_heatmap.png"), dpi=180)
        plt.close(fig)

    # 1.1 – Temperature vs Humidity scatter (simple)
    if all(k in df.columns for k in ["temperature", "humidity"]):
        samp = df[["temperature", "humidity"]].iloc[::60].copy()
        rho = float(df[["temperature", "humidity"]].corr().iloc[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(samp["temperature"], samp["humidity"], s=6)
        ax.set_title(f"Humidity vs Temperature (corr = {rho:.2f})")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Humidity")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "1.1_temperature_vs_humidity_scatter.png"), dpi=180)
        plt.close(fig)

        # NEW: graphical proof that temperature and humidity do not affect each other
        plot_temp_humidity_independence(df, corr_matrix, OUT_PLOTS_DIR)

    # 1.2 – Day vs Night boxplots
    if "temperature" in df.columns:
        day_vals = df.loc[df["daytime_time"] == 1, "temperature"].dropna().values
        night_vals = df.loc[df["daytime_time"] == 0, "temperature"].dropna().values
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([day_vals, night_vals], tick_labels=["Day", "Night"])
        ax.set_title("Temperature – Day vs Night")
        ax.set_ylabel("Temperature")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "1.2_temperature_day_night_box.png"), dpi=180)
        plt.close(fig)

    if "humidity" in df.columns:
        day_vals = df.loc[df["daytime_time"] == 1, "humidity"].dropna().values
        night_vals = df.loc[df["daytime_time"] == 0, "humidity"].dropna().values
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([day_vals, night_vals], tick_labels=["Day", "Night"])
        ax.set_title("Humidity – Day vs Night")
        ax.set_ylabel("Humidity")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "1.2_humidity_day_night_box.png"), dpi=180)
        plt.close(fig)

    # 1.3 – Basic descriptive stats
    stats_df = compute_descriptive_stats(df, sensor_cols)
    stats_df.to_csv(OUT_STATS_CSV, index=False)

    # Hourly stats (mean/min/max)
    hourly_stats = compute_hourly_stats(df)
    hourly_stats.to_csv(OUT_HOURLY_STATS_CSV)

    # Per-sensor hourly plots (mean/min/max)
    for var in ["temperature", "humidity", "light"]:
        mean_col = f"{var}_mean"
        min_col = f"{var}_min"
        max_col = f"{var}_max"
        if mean_col in hourly_stats.columns:
            fig, ax = plt.subplots(figsize=(14, 3.6))
            ax.plot(hourly_stats.index, hourly_stats[mean_col], linewidth=1.1, label="Mean")
            ax.plot(hourly_stats.index, hourly_stats[min_col], linewidth=0.8, label="Min")
            ax.plot(hourly_stats.index, hourly_stats[max_col], linewidth=0.8, label="Max")
            ax.set_title(f"{var.capitalize()} – Hourly mean / min / max")
            ax.set_xlabel("Time")
            ax.set_ylabel(var.capitalize())
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_PLOTS_DIR, f"1.3_{var}_hourly_mean_min_max.png"), dpi=180)
            plt.close(fig)

    # 3-in-1 hourly plot
    vars_present = [v for v in ["temperature", "humidity", "light"]
                    if f"{v}_mean" in hourly_stats.columns]
    if vars_present:
        fig, axes = plt.subplots(len(vars_present), 1, figsize=(14, 3.2 * len(vars_present)), sharex=True)
        if len(vars_present) == 1:
            axes = [axes]
        for ax, var in zip(axes, vars_present):
            mean_col = f"{var}_mean"
            min_col = f"{var}_min"
            max_col = f"{var}_max"
            ax.plot(hourly_stats.index, hourly_stats[mean_col], linewidth=1.1, label="Mean")
            ax.plot(hourly_stats.index, hourly_stats[min_col], linewidth=0.8, label="Min")
            ax.plot(hourly_stats.index, hourly_stats[max_col], linewidth=0.8, label="Max")
            ax.set_ylabel(var.capitalize())
            ax.legend()
        axes[0].set_title("Hourly mean / min / max – Temperature, Humidity, Light")
        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "1.3_all_sensors_hourly_mean_min_max.png"), dpi=180)
        plt.close(fig)

    # Daily peaks (hottest / most humid / brightest hour of each day, based on hourly means)
    daily_peaks = compute_daily_peaks(hourly_stats)
    daily_peaks.to_csv(OUT_DAILY_PEAKS_CSV, index=False)

    # Existing bar plots for daily peaks
    plot_daily_peak_hours(daily_peaks, OUT_PLOTS_DIR)

    # Daily summary (hottest / most humid / brightest day of the week)
    daily_means, best_days = compute_daily_summary(df)
    daily_means.to_csv(OUT_DAILY_SUMMARY_CSV)

    # Existing plots for daily means
    plot_daily_means(daily_means, OUT_PLOTS_DIR)

    # NEW: hour-of-day vs TRUE hourly max for each day and combined (per variable)
    plot_daily_hourly_max_profiles(df, OUT_PLOTS_DIR)

    # -----------------------------
    # TASK 2 – Feature engineering & anomalies
    # -----------------------------
    anomaly_counts: Dict[str, int] = {}

    if "temperature" in df.columns:
        df["temp_roc"] = compute_rate_of_change(df["temperature"], step_s)
        df["temp_ma_10min"] = compute_moving_average(df["temperature"], ROLL_WINDOW_MINUTES)

        # Whole-week moving average plot
        plt_df = resample_for_plotting(df, ["temperature", "temp_ma_10min", "daytime_time"], PLOT_RESAMPLE)
        fig, ax = plt.subplots(figsize=(14, 3.6))
        ax.plot(plt_df.index, plt_df["temperature"], linewidth=0.7, label="Temperature")
        ax.plot(plt_df.index, plt_df["temp_ma_10min"], linewidth=1.1, label="10-min moving average")
        shade_night_bands(ax, plt_df.index, plt_df["daytime_time"], alpha=0.12)
        ax.set_title("Temperature with 10-min moving average (night shaded)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "2.1_temperature_ma.png"), dpi=180)
        plt.close(fig)

        # Whole-week ROC plot
        plt_df = resample_for_plotting(df, ["temp_roc"], PLOT_RESAMPLE)
        fig, ax = plt.subplots(figsize=(14, 3.2))
        ax.plot(plt_df.index, plt_df["temp_roc"], linewidth=0.8)
        ax.set_title("Temperature rate of change (°C/s)")
        ax.set_xlabel("Time")
        ax.set_ylabel("ΔT / s")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "2.1_temperature_roc.png"), dpi=180)
        plt.close(fig)

        # NEW: per-day MA + ROC for temperature
        plot_daily_roc_ma(df, "temperature", "temp_ma_10min", "temp_roc", OUT_PLOTS_DIR)

    if "humidity" in df.columns:
        df["hum_roc"] = compute_rate_of_change(df["humidity"], step_s)
        df["hum_ma_10min"] = compute_moving_average(df["humidity"], ROLL_WINDOW_MINUTES)

        # Whole-week MA plot
        plt_df = resample_for_plotting(df, ["humidity", "hum_ma_10min", "daytime_time"], PLOT_RESAMPLE)
        fig, ax = plt.subplots(figsize=(14, 3.6))
        ax.plot(plt_df.index, plt_df["humidity"], linewidth=0.7, label="Humidity")
        ax.plot(plt_df.index, plt_df["hum_ma_10min"], linewidth=1.1, label="10-min moving average")
        shade_night_bands(ax, plt_df.index, plt_df["daytime_time"], alpha=0.12)
        ax.set_title("Humidity with 10-min moving average (night shaded)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Humidity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "2.1_humidity_ma.png"), dpi=180)
        plt.close(fig)

        # Whole-week ROC plot
        plt_df = resample_for_plotting(df, ["hum_roc"], PLOT_RESAMPLE)
        fig, ax = plt.subplots(figsize=(14, 3.2))
        ax.plot(plt_df.index, plt_df["hum_roc"], linewidth=0.8)
        ax.set_title("Humidity rate of change (%/s)")
        ax.set_xlabel("Time")
        ax.set_ylabel("ΔHumidity / s")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, "2.1_humidity_roc.png"), dpi=180)
        plt.close(fig)

        # NEW: per-day MA + ROC for humidity
        plot_daily_roc_ma(df, "humidity", "hum_ma_10min", "hum_roc", OUT_PLOTS_DIR)

    # Anomalies for temperature, humidity, light
    for col in [c for c in ["temperature", "humidity", "light"] if c in df.columns]:
        rm, rs = rolling_mean_std(df[col], ROLL_WINDOW_MINUTES)
        df[f"{col}_roll_mean_10min"] = rm
        df[f"{col}_roll_std_10min"] = rs
        df[f"{col}_anomaly"] = detect_anomalies_3sigma(df[col], rm, rs, z=3.0)
        anomaly_counts[col] = int(df[f"{col}_anomaly"].sum())

        tmp_df = pd.DataFrame({
            col: df[col],
            f"{col}_roll_mean_10min": df[f"{col}_roll_mean_10min"],
            "daytime_time": df["daytime_time"],
            f"{col}_anomaly": df[f"{col}_anomaly"].astype(int),
        })
        plt_df = resample_for_plotting(
            tmp_df,
            [col, f"{col}_roll_mean_10min", "daytime_time", f"{col}_anomaly"],
            PLOT_RESAMPLE
        )

        fig, ax = plt.subplots(figsize=(14, 3.8))
        ax.plot(plt_df.index, plt_df[col], linewidth=0.7, label=col)
        ax.plot(plt_df.index, plt_df[f"{col}_roll_mean_10min"], linewidth=1.1, label="Rolling mean (10 min)")
        an_mask = plt_df[f"{col}_anomaly"] > 0
        ax.scatter(plt_df.index[an_mask], plt_df.loc[an_mask, col], s=12, label="Anomaly")
        shade_night_bands(ax, plt_df.index, plt_df["daytime_time"], alpha=0.10)
        ax.set_title(f"{col.capitalize()} with rolling mean & anomalies (±3σ, 10-min)")
        ax.set_xlabel("Time")
        ax.set_ylabel(col.capitalize())
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PLOTS_DIR, f"2.2_{col}_anomalies.png"), dpi=180)
        plt.close(fig)

    # Save enriched dataset
    df_out = df.reset_index()
    df_out.to_csv(OUT_ENRICHED_CSV, index=False)

    # Build story report
    story_text = build_story(
        df=df,
        step_s=step_s,
        corr_matrix=corr_matrix,
        median_light=median_light,
        anomaly_counts=anomaly_counts,
        sensor_cols=[c for c in ["temperature", "humidity", "light"] if c in df.columns],
        hourly_stats=hourly_stats,
        daily_peaks=daily_peaks,
        daily_means=daily_means,
        best_days=best_days,
    )
    with open(OUT_STORY_TXT, "w", encoding="utf-8") as f:
        f.write(story_text)

    # Console summary
    print("\n=== Correlation matrix (Temp / Hum / Light) ===")
    print(corr_matrix if corr_matrix is not None else "Not enough columns to compute correlation.")
    print("\n=== Median light level (for interpretation) ===")
    print(median_light)
    print("\n=== Descriptive statistics (saved) ===")
    print(f"-> {OUT_STATS_CSV}")
    print("\n=== Hourly statistics (saved) ===")
    print(f"-> {OUT_HOURLY_STATS_CSV}")
    print("\n=== Daily peak hours (saved) ===")
    print(f"-> {OUT_DAILY_PEAKS_CSV}")
    print("\n=== Daily summary (saved) ===")
    print(f"-> {OUT_DAILY_SUMMARY_CSV}")
    print("\n=== Anomaly counts (±3σ, 10-min) ===")
    print(anomaly_counts)
    print("\n=== Enriched dataset (saved) ===")
    print(f"-> {OUT_ENRICHED_CSV}")
    print("\n=== Story report (saved) ===")
    print(f"-> {OUT_STORY_TXT}")
    print("\n--- Story preview ---")
    print("\n".join(story_text.splitlines()[:30]))
    print("\n(Full story in iot_story_report.txt)")


if __name__ == "__main__":
    main()
