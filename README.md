### Application description: 
The aim of the project was to create an application that allows to detect the most important variables in any data set that affect the result. The user sends a ready data set in CSV format, specifies what separator is used in the file. Then, they specify which column is the target column, and decides which columns will not be taken into account during the analysis. Finally, the user is shown a graph of variables that have the greatest impact on the previously selected target column. Finally, the user is shown a detailed description of the graph.



### Main functionalities:
* the user can send a file with data in CSV format, specifies the type of separator
* the user indicates the target column
* the user can decide which columns will not be taken into account during the analysis
* the application recognizes whether the sent data is related to a regression or classification problem. Then, it selects the appropriate model
* based on the model, a graph is displayed containing the most important variables that affect the result,
* Finally, the user is shown a detailed description of the graph

**To use the application, an OpenAI API key is required.**