# p318-Weather-Webcam
A project for CMPT 318

Uses machine learning to figure out weather data and times from webcam images.

The training data used is the provided scaled set of Kat Kam's images and the weather data from the Government of Canada.

Run this command to train the data and output a score:
```bash
python3 webcam_weather.py <csv weather data folder> <images folder> <target ID> <output folder>
```

The target ID determines the target for prediction. Input one of the following integers:

0 - Predict Weather at image date  
1 - Predict time of day (Early morning, Morning, Afternoon, or Evening)  
2 - Predict specific hour  
3 - Predict the next hour's weather

The images must have the datetime in the filename such as "katkam-20160605060000.jpg"

The only numbers in the filename should be the datetime.

The output folder specifies where a sample prediction plot will be saved at.

Here is an example command with the default katkam and yvr weather data:
```bash
python3 webcam_weather.py yvr-weather/ katkam-scaled/ 1 output
```
