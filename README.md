# Disaster Response Pipeline Project

## Table of Contents
- <a href="#1"> Description </a>
- <a href="#2"> Installation </a>
- <a href="#3"> Files structure </a>
- <a href="#4"> Licensing, Authors, Acknowledgements </a>

<a id='1'></a>
## Description

Thousands of messages are sent through diffrent means, yet it might be challenging to identify those that indicate a disaster. To filter out these messages and make sure only relevant disaster messages will be viewed, we will build a machine learning model that would predict relevant disaster messages. This project aims to create a web app that classify messages to specific disasters categories based on a model that was trained on real data. <br>

This reposiroty contains all files for Disaster Response Pipeline. There are 3 sections for building this project:
- ETL Pipeline
- ML pipeline 
- Web application

In the fisrt section (ETL pipeline) the data was prepared and merged and duplicates were removed then it was loaded in sql format. 
After that, ML pipeline was prepared which involved cleaning data, tokenaization, model selection and training, building a pipeline, then saving the trained model. Finally, to visulaze the daataset I've used the template provided by udacity to create graph using Plotly and to to classify messages based on the model I trained. 

<h3> Sample tests of the project: </h3>

<h4> Classifying a message </h4>



![sample 1](https://github.com/0xArwa/Disaster-response-pipeline/blob/main/images/36%20Disasters.png)


<h4> Graphs </h4>

![sample 2](https://github.com/0xArwa/Disaster-response-pipeline/blob/main/images/00%20Disasters.png)
![sample 3](https://github.com/0xArwa/Disaster-response-pipeline/blob/main/images/14%20Disasters.png)


<a id='2'></a>
## Installation 
You can run this project on your local machine through running the files in this order, process_data.py -> train_classifier.py -> run.py


Run the following command to store the dataset in database format 
<code>
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
</code>
Run the following command to train the model and save it 
<code>
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
</code>
Run the following command to deploy the web application 
<code>
python run.py 
</code>

<a id='3'></a>
## Project structure 

The files are organized in the follwoing structure: 

    app
    | - template
    | |- master.html # main page
    | |- go.html # final result
    |- run.py # Flask app
    data
    |- disaster_categories.csv # dataset
    |- disaster_messages.csv # dataset
    |- process_data.py #file for cleaning and formating data
    |- DisastorResponse.db # database
    models
    |- train_classifier.py #ML pipeline
    Images
    |- Images used for this README file
    README.md

<a id='4'></a>
## Licensing, Authors, Acknowledgements

The dataset belongs to Figure 8 and the project templates were provided by Udacity. 
