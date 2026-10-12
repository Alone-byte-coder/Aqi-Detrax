import pandas as pd

# --- 1. AQI CALCULATION FUNCTIONS ---

def get_sub_index(concentration, breakpoints):
    if pd.isna(concentration):
        return 0.0
    for low_c, high_c, low_i, high_i in breakpoints:
        if low_c <= concentration <= high_c:
            return ((high_i - low_i) / (high_c - low_c)) * (concentration - low_c) + low_i
            
    if concentration > breakpoints[-1][1]:
        return breakpoints[-1][3]
    return 0.0

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
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

def calculate_aqi_from_csv(file_path, pm25_col='PM2.5', pm10_col='PM10'):

    df = pd.read_csv(file_path)
    remove = ["Benzene","Toluene","Xylene","AQI","AQI_Bucket","City","NOx",]
    df = df.drop(columns=remove)
    df = df.dropna()    
    df['PM2.5_SubIndex'] = df[pm25_col].apply(lambda x: get_sub_index(x, PM25_BREAKPOINTS))
    df['PM10_SubIndex'] = df[pm10_col].apply(lambda x: get_sub_index(x, PM10_BREAKPOINTS))
    
    df['Final_AQI'] = df[['PM2.5_SubIndex', 'PM10_SubIndex']].max(axis=1)
    
    return df

def classify_aqi(aqi_value):
    if pd.isna(aqi_value):
        return "Unknown"
    elif aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"

if __name__ == "__main__":
 
    file_path = r"D:\project\BPUT\dataset\city_day_separated.csv"
    
    dataset = calculate_aqi_from_csv(file_path)
    dataset["Result"] = dataset["Final_AQI"].apply(classify_aqi)
    output_path = "D:\project\BPUT\dataset\output.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Successfully processed and saved to {output_path}")
    
    print(dataset[['PM2.5', 'PM10', 'Final_AQI', 'Result']].head())