# Traffic-Flow-Forecasting

Here is the big job modeling code repository for the YNU Big Data Analytics and Applications course.

### Get Started
```bash
git clone https://github.com/JackieLin2004/Traffic-Flow-Forecasting.git
cd ./Traffic-Flow-Forecasting
```

### Requirements
```bash
pip install -r requirements.txt
```

In this code repository, we used the PEMS04 and PEMS08 datasets for traffic flow prediction.

We not only used a spatial dimensional model for traffic flow prediction, but we also explored combining spatio-temporal features and employing a spatio-temporal synchronization approach for traffic flow prediction.

For spatio-temporally synchronized traffic flow prediction, we mainly use two models, `ASTGCN` and `MSTGCN`.

Here we give some indicator data:

<figure style="display: flex; align-items: center; justify-content: center;">
    <img src="./images/pems04_comparison.png" alt="">
</figure>

<figure style="display: flex; align-items: center; justify-content: center;">
    <img src="./images/pems04_difference_comparison.png" alt="">
</figure>

<figure style="display: flex; align-items: center; justify-content: center;">
    <img src="./images/pems08_comparison.png" alt="">
</figure>

<figure style="display: flex; align-items: center; justify-content: center;">
    <img src="./images/pems08_difference_comparison.png" alt="">
</figure>

### If you are interested in this project, feel free to fork and star!
