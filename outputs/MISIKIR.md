MISIKIR.md
# Summary of plots

## First day (2025-03-01)

### Patterns and trends

![Figure](outputs/plots/1.1 Minute average temerature versus time.png)
[Figure](outputs/plots/1.1 Minute average humidty versus time.png)
[Figure](outputs/plots/1.1 minute average light versus time.png)

Since the variation of sensor readings is changed every 5 seconds, it is difficult to understand and plot the trend of temperature, humidity and light intensity over time within that frequency. So, I have decided to plot and understand the minute average values of sensor readings to decrease the frequency of variations. According to the above plot the maximum average minute temperature was 23.8 ℃ recorded at 16:20:00 and the minimum minute average temperature was approximately 20.2 ℃ at 18:36:00. However, the maximum temperature throughout the day is 25 ℃ recorded at 00:36:00 and the minimum value of temperature was 20 ℃ at 00:14:25, which means both the maximum and minimum temperatures are recorded at midnight. In addition to this the maximum and minimum minute average humidity are recorded at 21:31:00 and 22:50:00 respectively. Furthermore, the maximum and minimum minute average light intensity values were also recorded at 07:59:00 and 08:11:00 respectively. Generally, the above graph tells us there is a huge fluctuation of temperature, humidity and light intensity readings over time which may be caused by external factors such as artificial light, sensor fault or dynamic weather environmental conditions.

![Figure](outputs/plots/1.1 Hourly average temerature versus time.png)
![Figure](outputs/plots/1.1 Hourly average light versus time.png)
![Figure](outputs/plots/1.1 Hourly average humidity versus time.png)

According to the hourly average values of sensor readings the maximum temperature value was recorded at 05:00 and the minimum value was recorded at 00:00. In case of humidity the maximum value was recorded at 06:00 and the minimum value was recorded at 18:00. Light intensity was also maximum at 11:00 and minimum at 10:00.

### Correlation

![Figure](outputs/plots/1.2 correlation matrix.png)

These results confirm the independence of temperature, humidity, and light in the monitored environment over the measured period.

None of the sensor readings are strongly influenced or linearly dependent on the others, suggesting either the environment is well-regulated or external factors do not meaningfully co-vary.

### Relationship between temperature and humidity

![Figure](outputs/plots/1.2 day and night cycle.png)
![Figure](outputs/plots/1.2 Relationship between temerature and humidity.png)

In practical terms, this plot demonstrates that within the measured range, temperature fluctuations did not cause humidity to rise or fall, and vice versa. This supports the idea that the environment is well-controlled or that external influences on humidity and temperature act independently.

The plot provides strong visual confirmation of a lack of any statistically significant relationship between temperature and humidity in this dataset. The values fluctuate independently, with no apparent pattern or predictive link between them.

### Min, max, mean and variance values

![Figure](outputs/plots/1.3 mean, min, max.png)
![Figure](outputs/plots/1.3 min, max, mean box plot.png)
![Figure](outputs/plots/1.3 variance.png)
![Figure](outputs/plots/1.3 Humidity over time max and min.png)
![Figure](outputs/plots/1.3 temerature over time max and min.png)
![Figure](outputs/plots/1.3 light over time max and min .png)

[`outputs/mis_1.csv`](outputs/mis_1.csv)


![Figure](outputs/plots/file.png)

The maximum light intensity value reached is approximately 999.96 units at 14:09:20 and the minimum value is around 100.04 at 02:25:45.

There is substantial variation in light readings, indicating active changes possibly due to daylight cycles or artificial lighting.

Temperature varies between a minimum of 20.00°C (at 00:14:25) and a maximum of 25.00°C (at 00:36:00), with frequent fluctuations, though always within a limited range. This suggests periodically but regulated changes potentially from climate control systems.

Humidity fluctuates between a minimum of 40.00% (at 00:32:10) and a maximum of 60.00% (at 23:00:10). It shows broad variation through the day, which may reflect changes in environmental conditions, ventilation, or occupancy.

### Anomaly detection and outlier detection

![Figure](outputs/plots/1.4 Temerature anomaly detection.png)
![Figure](outputs/plots/1.4 Humidity anomaly detection.png)
![Figure](outputs/plots/1.4 light intensity anomaly detection.png)

Condition: Flag anomaly if deviation > 2 × std from rolling mean.

There are 42 detected anomalies, indicating periods where humidity deviated significantly from typical levels.



37 anomalies are detected throughout the data, reflecting occasional abnormal spikes or drops in light readings. Most anomalies are at the extremes, close to the highest or lowest values, suggesting sudden changes in lighting conditions.



There are 34 detected anomalies, either sudden drops or spikes in temperature. Most temperature anomalies occur at the boundaries of the typical data range, suggesting quick transitions or possible sensor artifacts.



This visualization is useful to identify trends of frequent abnormal readings that may indicate sensor faults or extreme environmental events. Most anomalies are associated with extreme values (very high or low) that deviate from the normal rolling mean, potentially indicating environmental disruptions, abrupt changes, or sensor faults.

# FEATURES

![Figure](outputs/plots/1.5 Rate of change temerature versus humidity.png)
![Figure](outputs/plots/1.5 moving average comparsion.png)
![Figure](outputs/plots/1.5 rolling mean comparsion.png)


---

# WEEKLY DATASET

Plots to visualize the hourly variations of humidity (%), light intensity, and temperature over a week, from March 1 to March 8, 2025.


Trend and Fluctuations:

![Figure](outputs/plots/2.1 Hourly average temerature versus time.png)
![Figure](outputs/plots/2.1 Hourly average humidity versus time.png)
![Figure](outputs/plots/2.1 Hourly average light versus time.png)

All three variables (humidity, light intensity, temperature) show frequent short-term fluctuations, with values varying considerably from hour to hour. However, despite these fluctuations, none of the variables show a strong long-term upward or downward trend over the week; their average values remain relatively stable.

The hourly average humidity levels mostly oscillate between approximately 49.6% and 50.6%. Even though the variability is noticeable, there are no extreme peaks or persistent drops. This oscillation suggests a balance between environmental factors that add and remove moisture, possibly affected by indoor/outdoor transitions or ventilation cycles. Humidity is moderately variable, reflecting interactions between air, surfaces, and possibly ventilation.

The hourly average light intensity varies between roughly 530 and 570 lux. The fluctuations here might be more pronounced than those in humidity and temperature, possibly reflecting changes in daylight, artificial lighting, or room usage patterns. There is no strong day/night pattern evident, but there are unpredictable peaks and troughs that may align with specific events or periods of increased activity. Light intensity exhibits the widest short-term changes, likely due to external factors (windows, usage patterns) or deliberate light adjustments.

The hourly average temperature values are clustered tightly, ranging from about 22.35°C to 22.60°C. These readings imply a highly regulated thermal environment, typical of an indoor setting with controlled heating or cooling since temperature is most stable, suggesting active regulation. Fluctuations are present, but the overall spread is the narrowest, indicating minimal external influence or variability.

Generally, these plots tell us that in a controlled environment, temperature remains highly stable, humidity experiences moderate natural variation, and light intensity is subject to frequent and sometimes large fluctuations, probably driven by human activity and external conditions.

![Figure](outputs/plots/2.1 Daily average temerature over time.png)
![Figure](outputs/plots/2.1 Daily average humidity versus time.png)
![Figure](outputs/plots/2.1 Daily average light over time.png)

All parameters exhibit small, gradual changes over the week, with no sudden dramatic shifts. Temperature and humidity generally remain stable with minor daily variation, suggesting effective environmental control likely in an indoor or regulated space. However, light intensity has more pronounced daily swings compared to the other parameters, particularly the sharp peak on March 5, possibly due to external events or changes in lighting habits.

According to the daily average plot the hottest day is March 7, 2025, and the coolest day is March 4, 2025.

### Correlation 

![Figure](outputs/plots/2.2 correlation matrix.png)

1. **Humidity–Temperature Inverse Relationship Over Time**

![Figure](outputs/plots/2.2 temerature and humidity inverse relationship - Copy.png)
There is no clear visual evidence of a strong inverse (negative) relationship between their trends; they appear largely independent.

2. **Day Night Light Cycles**  

![Figure](outputs/plots/2.2 day and night cycle.png)
Light intensity is high during daytime and drops in the shaded periods, visually distinguishing between day and night cycles. However, the classification does not clearly show the regular day and night cycle period due to other external factors which affect light intensity.

![Figure](outputs/plots/file.png)

All three sensors report values that remain inside well-defined, narrow ranges over time. Temperature, humidity, and light intensity do not show strong correlation or mutual influence either visually or mathematically.

### Basic statistics

![Figure](outputs/plots/2.3 mean, min, max.png)

![Figure](outputs/plots/2.3 box plot of min, max, mean - Copy.png)

---

# 1.4 Anomalies

![Figure](outputs/plots/2.4 temerature anomaly detection.png)
![Figure](outputs/plots/2.4 Humidity anomaly detection.png)
![Figure](outputs/plots/2.4 light intensity anomaly detection.png)
---

# Features

### Moving Average: Temperature vs Humidity

![Figure](outputs/plots/2.5 average moving rate comparsion.png)

The temperature moving average of a week remains steady around 22°C–23°C, indicating highly stable indoor climate control. Humidity moving average consistently stays near 50%, also showing environmental stability. Both variables display minor short-term fluctuations but no large shifts over the week.

### Rate of Change: Temperature vs Humidity

![Figure](outputs/plots/2.5 rate of change of temerature versus humidity.png)

The values oscillate around zero, meaning that at most times, changes are gradual and balanced rather than abrupt. The spread (height of oscillation) indicates some variability, but since the mean rate remains near zero, the environment is not experiencing sustained increasing or decreasing trends.

### Rolling Mean Comparison: Temperature, Humidity, Light

![Figure](outputs/plots/2.5 rolling mean comparsion.png)

This plot expands the comparison by including light intensity alongside temperature and humidity rolling means. Light intensity’s rolling mean is around 500–600 units, showing more noticeable variation than temperature or humidity.

Temperature and humidity remain stable and nearly flat over time, reinforcing the interpretation of well-controlled indoor conditions.
