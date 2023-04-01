# Disaster Response Pipeline Project

## Table of Contents
- <a href="#1"> Description </a>
- <a href="#2"> Installation </a>
- <a href="#3"> Licensing, Authors, Acknowledgements </a>

<a id='1'></a>
## Description
This reposiroty contains all files for Disaster Response Pipeline. There are 3 sections for building this project:
- ETL Pipeline
- ML pipeline 
- Web application

In the fisrt section (ETL pipeline) the data was prepared and merged and duplicates were removed then it was loaded in sql format. 
After that, ML pipeline was prepared which involved cleaning data, tokenaization, model selection and training, building a pipeline, then saving the trained model. Finally, to visulaze the daataset I've used the template provided by udacity to create graph using Plotly and to to classify messages based on the model I trained. 

Sample tests of the project: 

Classifying a message



![sample 1](https://github.com/0xArwa/Disaster-response-pipeline/blob/main/images/36%20Disasters.png)
-- picture here -- 

Graphs 
![sample 2](https://github.com/0xArwa/Disaster-response-pipeline/blob/main/images/00%20Disasters.png)
![sample 3](https://github.com/0xArwa/Disaster-response-pipeline/blob/main/images/14%20Disasters.png)


<a id='2'></a>
## Installation 
You can run this project on your local machine through running the files in this order, process_data.py -> train_classifier.py -> run.py



<a id='3'></a>
## Licensing, Authors, Acknowledgements

The dataset belongs to Figure 8 and the project templates were provided by Udacity. 
