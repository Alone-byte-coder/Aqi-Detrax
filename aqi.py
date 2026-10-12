
PM25_BREAKPOINTS = [
    (0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500)
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500)
]


def get_sub_index(value, breakpoints):
    for low_c, high_c, low_i, high_i in breakpoints:
        if low_c <= value <= high_c:
            return round(
                (high_i - low_i) / (high_c - low_c)
                * (value - low_c) + low_i,
                2
            )

    if value > breakpoints[-1][1]:
        return 500.0

    # Handle small gaps between breakpoint ranges.
    for i in range(len(breakpoints) - 1):
        current = breakpoints[i]
        following = breakpoints[i + 1]

        if current[1] < value < following[0]:
            return float(
                current[3]
                if value - current[1] <= following[0] - value
                else following[2]
            )

    return 0.0


def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def calculate_aqi(pm25, pm10):
    pm25_index = get_sub_index(pm25, PM25_BREAKPOINTS)
    pm10_index = get_sub_index(pm10, PM10_BREAKPOINTS)

    final_aqi = max(pm25_index, pm10_index)

    return {
        "pm25_sub_index": pm25_index,
        "pm10_sub_index": pm10_index,
        "final_aqi": final_aqi,
        "category": classify_aqi(final_aqi)
    }
