# dappi
Semantic search API for Dapper content

# Get Started
## Using Docker
A .env will be needed use .envtemplate for complete your own enviroment variables in other case if you need yours to interact with Dapper ask it.

run compose.yaml using:
```
docker compose -d
```

if Dockerfile was not edited the API will response at 8080 port

## By your own
Requirements
| Dependency | version       |
|------------|---------------|
| chromadb   | lastest       |
| python     | python:3.11.5 |

Once you created your enviroment install all packages needed using:
```
pip install -r requirements.txt
```
>This action can take some minutes due to langchain size we are working for only import strictly packages needed then less time!

### Windows
For that case you will need install pywin32 cause it is not included in requirements.txt
```
pip install pywin32
```

#Update vectorestore
Call enpoint GET /Update in order to get all data from database 
```
localhost:8080/update
```

#Getting Answers
Using GET ask/{question} you will get JSON with a short answer for you question including content related that was used to answer you
```
localhost:8080/ask/answer%20here
```
