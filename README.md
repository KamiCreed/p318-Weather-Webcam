# p318-Weather-Webcam
A project for CMPT 318

Uses machine learning to figure out weather data from webcam images.

The training data used is Kat Kam's images and the provided weather data.

Run this command to train the data and output a score:
```bash
python3 webcam_weather.py <csv weather data folder> <images folder> <target ID>
```

The target ID determines the target for prediction. Input one of the following integers:

0 - Predict Weather at image date
1 - Predict time of day (Early morning, Morning, Afternoon, or Evening)
2 - Predict specific hour
3 - Predict the next hour's weather

The images must have the datetime in the filename such as "katkam-20160605060000.jpg"

The only numbers in the filename should be the datetime.