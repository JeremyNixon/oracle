import time
import pandas as pd

def time_preprocessing(df, column_name):
    
    times = []
    count = 0
    start = time.time()
    for i in df[column_name]:
        count += 1
        if count % 100000 == 0:
            time_elapsed = time.time() - start
            #print "Count = %r, Time Elapsed = %r" %(count, time_elapsed)
        times.append(time.strptime(i, "%Y-%m-%d %H:%M:%S"))

    year = []
    month = []
    day = []
    hour = []
    minute = []
    second = []
    day_of_week = []
    day_in_year = []

    for i in times:
        year.append(i[0])
        month.append(i[1])
        day.append(i[2])
        hour.append(i[3])
        minute.append(i[4])
        second.append(i[5])
        day_of_week.append(i[6])
        day_in_year.append(i[7])

    df['year'] = year
    df['month'] = month
    df['day'] = day
    df['hour'] = hour
    df['minute'] = minute 
    df['second'] = second
    df['day_of_week'] = day_of_week
    df['day_in_year'] = day_in_year
    df.drop([column_name], axis=1)
    return df