---
interact_link: content/machine-learning/aws/aws-big-data/aws-big-data-specialty-analysis.ipynb
kernel_name: python3
has_widgets: false
title: 'Analysis'
prev_page:
  url: /machine-learning/aws/aws-big-data/aws-big-data-specialty-exam
  title: 'AWS Big Data Specialty Exam'
next_page:
  url: /machine-learning/aws/aws-big-data/aws-big-data-specialty-collection
  title: 'Collections'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# AWS Big Data Specialty Exam Notes

Requirements:
1. Order history App
    1. Server logs 
    2. Amazon Kinesis Data Streams 
    3. AWS Lambda 
    4. Amazon DynamoDB
    5. Client App
2. Product Recommendations
    1. Server logs 
    2. Amazon Kinesis Data Firehose
    3. Amazon S3
    4. Amazon EMR
3. Predicting order quantities
    1. Server logs
    2. Amazon Kinesis Data Firehose
    3. Amazon S3
    4. Amazon ML / Sagemaker
4. Transaction Rate Alarm
    1. Server logs
    2. Amazon Kinesis Data Streams
    3. Amazon Kinesis Data Analytics
    4. Amazon Kinesis Data Streams
    5. AWS Lambda
    6. Amazon SMS
5. Near-real-time log analysis
    1. Server logs
    2. Amazon Kinesis Data Firehose
    3. Amazon Elasticsearch Service
6. Data warehousing & visualization
    1. Server logs
    2. Amazon Kinesis Data Firehose
    3. Amazon S3
        - Serverless
            1. AWS Glue
            2. Amazon Athena
        - Managaged
            1. Amazon Redshift
            2. Amazon QuickSight



---
# Kinesis Analytics
- Similar to Spark Streaming, but specific to Kinesis
- Can use Firehose or Data Streams as an input source but also files in S3
- Application code in Kinesis Analytics will then process the stream and output the data to either Streams or Firehose and subsequently to S3 or Redshift
- An in-application error stream will be automatically provided for every application

Use Cases:
- Streaming etl
    - E.g. Process data to a specific format / schema and deliver that data to S3
- Continuous Metric generation
    - E.g. Building a live leaderboard for a mobile game by computing the top players every minute and sending that to DynamoDB or check traffic to website by calculating number of unique web site visitors every 5min and sending results to Redshift for further analysis
- Responsive Analytics
    - E.g. Application to look for events that meet certain criteria and automatically notify right customers using Kinesis data streams and SNS
    
Cost Model:
- Pay for resources consumed but expensive
- Serverless
- Use IAM permissions to access streaming source
- Schema discovery
    - Like what Glue Crawler does
    
Random Cut Forest:
- SQL function used for anomaly detection



---
# Requirement 4: Transaction rate alarm
Configure Kinesis Analytics to having a sliding window of 10s and check if there are more than 10 orders in that 10s window which is what we'll define as an anomaly

1. Create new Data Stream that will send the Kinesis Analytics data about whether there's an anomaly to AWS Lambda (Lambda polls from the stream actually)
    1. Click `Create data stream`
    2. name: `OrderRateAlarms`
    3. number of shards: `1` and create stream

2. Create Kinesis Analytics application
    1. Go to `Data Analytics` and click on `Create application`
    2. name: `TransactionRateMonitor` and create application
    3. `ConnectStreamingData` and choose `CadabraOrders` Kinesis data stream
    4. Run `sudo ./LogGenerator.py` and click `Discover Schema`
    5. `Go to SQL Editor` under Real time analytics and insert the code below into the IDE and `Save and run SQL`
    6. Connect to a destination
        - Click `Destination` below and `Connect to a destination`
        - We can only choose to send it to either Firehose or Streams, and because we want this to be real-time, we'll connect it to another data stream, the one we just created, `OrderRateAlarms`
        - Choose `In-application stream` to be `TRIGGER_COUNT_STREAM`, one of the tables created by the SQL sxript below
        - Ouput format: json, Create / Update IAM role ..., save and continue

3. Now that we've created a new Kinesis stream that's receiving data whenever we're in an alarmed state at most once per minute, we need to trigger a lambda function that will do something with that data, in this case it would be to call AWS SNS to send us a notification
    1. Create new IAM role for Lambda function
        - Go to IAM console
        - Click on `Roles` and `Create Role`
        - Click for `Lambda` service, Next:Permissions
        - Search and tick 
            1. `AWSLambdaKinesisExecutionRole`
            2. `AmazonSNSFullAccess`
            3. `CloudWatchLogsFullAccess`
            4. `AWSLambdaBasicExecutionRole`
        - Skip Tags
        - Role name: `LambdaKinesisSNS` and `Create role`
    2. Create Lambda Function
        - Go to `Lambda > Functions > Create function`
        - `Author from scratch`, name: `TransactionRateAlarm`
        - Runtime: `Python 2.7` and choose role: `LambdaKinesisSNS`
        - `Create function`
        - Click on `Kinesis` on left hand side menu `Designer`
        - Select `OrderRateAlarms` to be the Kinesis stream to listen for updates on and `Add`
        - Configure lambda function itself
            - Change `Timeout` in `Basic settings` to `1min` instead
            - Add the code snippet below into IDE
                - Create SNS topic
                    - Go to SNS console, `Create topic`
                    - topic name: `CadabraAlarms` and display name: `Cadabra` and `Create topic`
                    - Create subscription on the topic (What it'll do when it gets fired)
                        - Protocol: `SMS`
                        - Endpoint: `YOUR_CELL_PHONE_NUMBER` and `Create subscription`
                    - `Publish topic`, subject: `test`, message: `this is a test` and `Publish message`
                    - Copy `topic ARN` (ARN - Amazon Resource Name)
            - Paste the topic ARN into the code and `Save`
    
### analytics-query.txt
```mysql
CREATE OR REPLACE STREAM "ALARM_STREAM" (order_count INTEGER);

CREATE OR REPLACE PUMP "STREAM_PUMP" AS 
    INSERT INTO "ALARM_STREAM"
        SELECT STREAM order_count
        FROM (
            SELECT STREAM COUNT(*) OVER TEN_SECOND_SLIDING_WINDOW AS order_count
            FROM "SOURCE_SQL_STREAM_001"
            WINDOW TEN_SECOND_SLIDING_WINDOW AS (RANGE INTERVAL '10' SECOND PRECEDING)
        )
        WHERE order_count >= 10;

CREATE OR REPLACE STREAM TRIGGER_COUNT_STREAM(
    order_count INTEGER,
    trigger_count INTEGER);
    
CREATE OR REPLACE PUMP trigger_count_pump AS INSERT INTO TRIGGER_COUNT_STREAM
SELECT STREAM order_count, trigger_count
FROM (
    SELECT STREAM order_count, COUNT(*) OVER W1 as trigger_count
    FROM "ALARM_STREAM"
    WINDOW W1 AS (RANGE INTERVAL '1' MINUTE PRECEDING)
)
WHERE trigger_count >= 1;
```

### lambda.txt
```python
from __future__ import print_function
import boto3
import base64

client = boto3.client('sns')
# Include your SNS topic ARN here.
topic_arn = '<your topic here>'

def lambda_handler(event, context):
    try:
        client.publish(TopicArn=topic_arn, Message='Investigate sudden surge in orders', Subject='Cadabra Order Rate Alarm')
        print('Successfully delivered alarm message')
    except Exception:
        print('Delivery failure')
```



---
# Kinesis Analytics Quiz



1. From which sources can the input for Kinesis analytics be obtained ?
    - Kinesis Analytics can only monitor streams from Kinesis, but both data streams and Firehose are supported.



2. After real-time analysis has been performed on the input source, where may you send the processed data for further processing?
    - While you might in turn connect S3, Redshift, or Lambda to your Kinesis Analytics output stream, Kinesis Analytics must have a stream as its input, and a stream (Data Stream or Firehose) as its output.



3. If a record arrives late to your application during stream processing, what happens to it?
    - The record is written to the error stream when the record arrives past the Timeout we set



4. You have heard from your AWS consultant that Amazon Kinesis Data Analytics elastically scales the application to accommodate the data throughput. What though is default capacity of the processing application in terms of memory?
    - 32GB = 4GB/KPU * 8KPUs
    - Kinesis Data Analytics provisions capacity in the form of Kinesis Processing Units (KPU). A single KPU provides you with the memory (4 GB) and corresponding computing and networking. The default limit for KPUs for your application is eight.



5. You have configured data analytics and have been streaming the source data to the application. You have also configured the destination correctly. However, even after waiting for a while, you are not seeing any data come up in the destination. What might be a possible cause?
    - Issue with IAM role, Mismatched name for output stream, Destination service currently unavailable (Data stream / Firehose is busy working with another producer)



---
# Amazon Elasticsearch Service (ES)

- Petabyte scale analysis and reporting, started out as a search engine
- Elastic Stack
    - Documents
    - Types (Deprecated)
    - Index
        - Every document is hashed to a shard which live in different nodes within a cluster
        - Every shard is actually a self-contained lucene index
        - Write requests are routed to the primary node then replicated
        - Read requests are routed to any primary or replica node
- Not serverless, we need to decide how many servers like EMR
- Have to decide upfront if cluster wants to be in VPC, cannot move in and out of VPC after cluster is launched
    - If we are using VPC but need to use Kibana (only accessible through the web interface), we can use AWS Cognito to allow end users to log into kibana through an enterprise identity provider such as Microsoft active directory using saml 2.0 and also through social identity providers such as Google, Facebook, Amazon.
        - People can then log into Cognito using their Facebook account and that will grant them access to get into Kibana even if it's hidden behind a VPC

Anti-patterns
- OLTP
    - Use RDS / DynamoDB better
- Ad-hoc data querying
    - Use Athena instead because ES is primarily for search and analytics



---
# Requirement 5: Near-real-time Log Analysis

Send real Apache Server logs into Firehose and then stream it to Elasticsearch so that we can interactively search and create dashboards / visualizations using Kibana on top of raw server data

1. Create Server logs in EC2 instance
    1. Go to EC2 instance in Terminal
    - `wget http://media.sundog-soft.com/AWSBigData/httpd.zip`
    - `unzip httpd.zip`
    - `cd httpd` and `less ssl_access_log`
    - `cd ~` and move the data into `/var/log/httpd` by `sudo mv httpd /var/log/httpd`
    
2. Create Elasticsearch cluster to listen to this data
    1. Go to Elasticsearch console
    - `Create domain` (This is what they call a database in Elasticsearch)
    - Deployment type: `Development and testing` because we don't want to spend a lot of money on multiple availability zones
    - Elasticsearch version: `6.4` and `Next`
    - domain name: `cadabra`
    - instance type: `m4.large.elasticsearch`
    - number of instances: `1`
    - Don't enable dedicated master instance
    - EBS for storage is fine
    - We don't need encryption but it's good to understand that we have encryption options - node to node / data at rest
    - `Next`
    - Normally we want to keep cluster in a VPC, but because we need a VPN in order to access KIbana, we'll make this `Public access`
    - We have the option to `Enable Amazon Cognito for authentication` so that other people can actually log in to use this Kibana instance 
    - For the access policy, we want to restrict usage of Kibana to only this account, so go to `Account Settings` and copy `Account Id`
        - Set the domain access policy to: `Allow or deny access to one or more AWS accounts or IAM users` and paste in `Account ID or ARN`
        - `Next`
    - `Confirm`
    
3. `Create Firehose Delivery Stream`
    1. `Create Delivery Stream`
    - name: `WebLogs`
    - source: `Direct PUT` because we're going to use the Kinesis agent as producer to flow the server log data into firehose
    - `Process records`
        - We need to transform the source records here because if we try to transform the Apache Logs on the client side using the Kinesis Agent, the timestamps dont end up in the right format for Elasticsearch
            - Create Lambda function to handle the transformation from raw apache logs to JSON data that Elasticsearch accepts
                - Go to `Transform source records with AWS Lambda` and set record transformation: `Enabled` and `Create New` lambda function
                    - Click `Apache Log to JSON` template 
                    - Name: `LogTransform`
                    - Role: `Create a custom role`
                        - `Allow`
                - `Create function`
                - Scroll down and change timeout: `1min` and `Save`
        - Go back to `Process records` and choose `LogTransform` as the lambda function and `Next`
    - `Choose destination`
        - `Elasticsearch service` (Remember that Firehose can only stream to S3, Redshift, Elasticsearch, Splunk)
        - domain: `cadabra`
        - index: `weblogs`
        - index rotation: `Every day` so that it's very easy and efficient to drop old data
        - type: `weblogs`
        - S3 Backup:
            - backup mode: `Failed records only`
            - Backup S3 bucket: `orderlogs-sundogedu`
            - backup s3 bucket prefix: `es/`
            - `Next`
    - `Configure settings`
        - Specify buffer conditions
            - Buffer size:  `5MB`
            - Buffer interval: `60s`
        - Create new IAM role
            - Defaults are good, so `Allow`
    - Ensure that Elasticsearch is actually running before we finish creating the firehose stream and `Create delivery stream`
    
4. Configure Kinesis agent to pick up the logs and put them into the firehose stream
    1. `sudo nano /etc/aws-kinesis/agent.json`
    2. Add new flow by inserting the following:
        - `"filePattern": "/var/log/httpd/ssl_access*",`
        - `"deliveryStream": "WebLogs",`
        - `"initialPosition": "START_OF_FILE"` so that we don't have to manually keep pushing data into there and pick things up from the beginning of the data that I already copid in there
        - CTRL + O, Enter, CTRL + X
    3. Restart agent by `sudo service aws-kinesis-agent restart`
    4. See logs `tail -f /var/log/aws-kinesis-agent/aws-kinesis-agent.log`
    5. Go back to Elasticsearch console and see `My domains > cadabra > Indices`
        - See that the count for the day is not 0
        
5. Grant access from our desktop to this ES cluster so that we can use Kibana
    - In `My domains > cadabra`, click on `Modify access policy`
    - Add another clause under access policy that opens up Kibana from our IP address
        - Paste the code snippet below into the access policy under `Statement` list and replace `<your IP>` with your actual IP address and `<your account id>` with your actual account id
        - `Submit`
        - Click the link to Kibana `Overview` 

6. Explore Kibana
    - Go to `Management > Index Pattern` and `Create index pattern`
        - index pattern: `weblogs*` and `Next step`
        - time filter field name: `@timestamp` which is a local time
        - `Create index pattern`
    - `Discover`
        - Change time range so that we can see an absolute time range instead of the last 15 min
            - Click on the time range on top right hand corner and change it to an absolute range, `2019-01-27` to `2019-02-02`
        - We can now search for `response:500`
            - We see that there's a big peak of 11 of these "Internal Server Error"s so it would indicate that this should be something we have to investigate further as to what caused it.
    - Go to `Visualize`
        - We can create a `VerticalBar` chart
            - Select `weblogs*` index and we can visualize anything we want now
    
```json
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "*"
      },
      "Action": [
        "es:*"
      ],
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": [
            "<your IP>"
          ]
        }
      },
      "Resource": "arn:aws:es:us-east-1:<your account id>:domain/cadabra/*"
    },
```



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
!wget http://media.sundog-soft.com/AWSBigData/httpd.zip

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
--2019-06-15 10:54:39--  http://media.sundog-soft.com/AWSBigData/httpd.zip
Resolving media.sundog-soft.com (media.sundog-soft.com)... 52.216.97.43
Connecting to media.sundog-soft.com (media.sundog-soft.com)|52.216.97.43|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 39403376 (38M) [application/octet-stream]
Saving to: 'httpd.zip'

httpd.zip           100%[===================>]  37.58M   446KB/s    in 2m 11s  

2019-06-15 10:56:51 (293 KB/s) - 'httpd.zip' saved [39403376/39403376]

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
!unzip httpd.zip

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Archive:  httpd.zip
   creating: httpd/
  inflating: httpd/access_log        
  inflating: httpd/access_log-20190120  
  inflating: httpd/access_log-20190127  
  inflating: httpd/access_log-20190203  
  inflating: httpd/access_log-20190210  
  inflating: httpd/error_log         
  inflating: httpd/error_log-20190120  
  inflating: httpd/error_log-20190127  
  inflating: httpd/error_log-20190203  
  inflating: httpd/error_log-20190210  
  inflating: httpd/ssl_access_log    
  inflating: httpd/ssl_access_log-20190120  
  inflating: httpd/ssl_access_log-20190127  
  inflating: httpd/ssl_access_log-20190203  
  inflating: httpd/ssl_access_log-20190210  
  inflating: httpd/ssl_error_log     
  inflating: httpd/ssl_error_log-20190120  
  inflating: httpd/ssl_error_log-20190127  
  inflating: httpd/ssl_error_log-20190203  
  inflating: httpd/ssl_error_log-20190210  
  inflating: httpd/ssl_request_log   
  inflating: httpd/ssl_request_log-20190120  
  inflating: httpd/ssl_request_log-20190127  
  inflating: httpd/ssl_request_log-20190203  
  inflating: httpd/ssl_request_log-20190210  
```
</div>
</div>
</div>



---
# Elasticsearch Quiz



1. How can you ensure maximum security for your Amazon ES cluster?
    - Bind with a VPC, Use security groups, Use IAM policies, Use access policies associated with the Elasticsearch domain creation



2. As recommended by AWS, you are going to ensure you have dedicated master nodes for high performance. As a user, what can you configure for the master nodes?
    - Only the count and instance types of the master nodes, not EBS volume associated with the node or the upper limit of network traffic / bandwidth



3. Which are supported ways to import data into your Amazon ES domain?
    - Kinesis, DynamoDB, Logstash / Beats, and Elasticsearch's native API's offer means to import data into Amazon ES.



4. What can you do to prevent data loss due to nodes within your ES domain failing?
    - Maintain snapshots of ES domain
    - Amazon ES created daily snapshots to S3 by default, and you can create them more often if you wish.



5. You are going to setup an Amazon ES cluster and have it configured in your VPC. You want  your customers outside your VPC to visualize the logs reaching the ES using Kibana. How can this be achieved?
    - Use a reverse proxy, VPN, or VPC Direct Connect



---
# Amazon Athena
- petabyte scale
- SQL Engine for doing interactive queries on data in an S3 data lake and completely serverless
- Supported data formats:
    - CSV, JSON (Human Readable)
    - ORC, Parquet, Avro (Columnar except for Avro, splittable - can be distributed across nodes in a cluster)
- Use cases:
    - Ad-hoc queries of weblogs (Better to use Athena than Elasticsearch for project above)
- Use columnar format to save money on athena
- Dont use this for Highly formatted reports and ETL (Use glue instead)



---
# Requirement 6: Data warehousing & visualization

We want to connect Glue to crawl the S3 bucket with order logs that have been saved inside previously using Firehose. We can then query that data using SQL from Athena.

1. Configure Glue
    1. Go to AWS Glue console
    - `Crawlers` and `Add a crawler`
    - name: `order data`
    - we can encrypt logs from Glue that will be pushed to CloudWatch but this isnt important right now
    - Choose data source: `S3`, we have `JDBC` and `DynamoDB` as options too
        - Choose S3 Path: `orderlogs-sundogedu`
        - Exclude any unecessary data, `Exclude patterns`
            - `es/**` to exclude all the errors we stored in the same S3 bucket for a previous exercise
            - `Next`
    - Create IAM Role:
        - AWSGlueServiceRole-`OrderData` and `Next`
        - Frequency: `Run on demand` because our schema wont change any time soon
        - Add database to store the structured S3 data so we can run sql queries on it with Athena
            - name: `orderlogs` and `Next`
    - `Finish`
    - `Run it now`
    - Go to `Databases` and click `orderlogs`
    - Click `tables in orderlogs`
    - `Edit schema` to add the column names that arent present in the CSV
        - col1: `InvoiceNo`
        - col2: `StockCode`
        - col3: `Description`
        - col4: `Quantity`
        - col5: `InvoiceDate`
        - col6: `UnitPrice`
        - col7: `CustomerID`
        - col8: `Country`
        - partition0: `year`
        - partition1: `month`
        - partition2: `day`
        - partition3: `hour`
    - If the data types aren't right, best is to create an ETL job in Glue and write a script to convert, dont amend it manually in `Edit schema`
    - `Save`
    
- Go to Athena and try to query this table created from the S3 data lake
    - Go to Athena console and set Database: `orderlogs`
    - In new query, type: 
```sql 
SELECT description, count(*) 
from orderlogs_sundogedu 
where country='France' and year='2019' and month='02' 
group by description
```



---
# Athena Quiz



1. As a Big Data analyst, you need to query/analyze data from a set of CSV files stored in S3. Which of the following serverless services helps you with this?
    - AWS Athena



2. What are two columnar data formats supported by Athena?
    - Parquet and ORC



3. Your organization is querying JSON data stored in S3 using Athena, and wishes to reduce costs and improve performance with Athena. What steps might you take?
    - Convert JSON data to ORC and analyze ORC data with Athena
    - Using columnar formats such as ORC and Parquet can reduce costs 30-90%, while improving performance at the same time



4. When using Athena, you are charged separately for using the AWS Glue Data Catalog. True or False ?
    - True



5. Which of the following statements is NOT TRUE regarding Athena pricing?
    - Amazon Athena charges you for cancelled queries [TRUE]
    - Amazon Athena charges you for failed queries [FALSE]
    - You will get charged less when using a columnar format [TRUE]
    - Amazon Athena is priced per query and charges based on amount of data scanned by the query [TRUE]



---
# Amazon Redshift
- exabyte scale data warehouse
- scale up /  down on demand
- spectrum is just like athena, creates tables without copying the data from S3 so that you can query it
- Import / Export data
    - Most efficient way to import data is by using `COPY` command
        - Can read from multiple data files / data streams simultaneously
            - S3, EMR, DynamoDB, some remote host using SSH
    - Can use rol-based / key-based access control to provide authentication to do `COPY` command
        - E.g. `COPY` data from S3 Bucket using `copy <table name> <authorization: you can use either IAM followed by ARN or access key id followed by access key id secret access key followed by secret access key>` OR use manifest JSON file sitting in S3 that lists data files  you want to load then `copy <table name> <manifest file> <authorization>`
        - Export data by using `UNLOAD` command into S3



---
# Requirement 6: Data warehousing & Visualization

We'll connect Glue after the configuration in the previous exercise so that we can simply use Redshift spectrum on top of the `orderlogs` database created by Glue.

1. Configure Redshift cluster
    1. Go to redshift console
    - `Clusters > Launch cluster`
    - cluster identifier: `cadabra`
    - database name: `dev`
    - port: `5439`
    - master user name: `awsuser` (anything here)
    - `Continue` and skip node config because default is fine
    - We can encrypt the database using KMS / HSM, but we'll stick with none for now and stick with default VPC
    - Create security group for VPC 
        1. Go to VPC console
        - Go to `security groups` in side menu and `Create security group`
        - Security group name: `RedshiftSecurity`
        - description: `Permissions for redshift`
        - vpc: `<choose your default vpc>` and `Create`
        - Scroll down to `Inbound Rules`
            - Set inbound rule so that we can open up ports 22 and 5439 (Redshift runs on port 5439 and SSH is supported at 22) - Notice that port 22 isn't here so we can't connect through Terminal using SSH
        - Input `Custom TCP Rule`, `22`, `My IP`, and click `Save rules`
        - Input `Custom TCP Rule`, `5439`, `My IP`, and click `Save rules` and `Close`
    - We need an IAM role too
        1. Go to IAM console
        - `Create role`
        - select type of trusted entity: `AWS service`
        - choose service that will use this role: `Redshift`
        - select your use case: `Redhsift customizable`
        - `Next Permissions`
        - Policies:
            - `AmazonS3ReadOnlyAccess`
            - `AWSGlueConsoleFullAccess`
        - Skip `Next Tags`
        - role name: `RedshiftSpectrum`
        - `Create role`
    - VPC security groups: `RedshiftSecurity (...)`
    - Available IAM roles: `RedshiftSpectrum`
    - `Continue` and `Launch Cluster`
    
2. Query
    1. Go to `Query editor`
    - cluster: `cadabra`
    - database: `dev`
    - database user: `awsuser`
    - password: `<insert your own password>`
    - `Connect`
    - Get ARN for IAM role (`RedshiftSpectrum`) we just created
    - Enter code snippet below in new query 1 and run the query
    - Change schema to `information_schema` 
    - Run the query below
 
Bring data from S3 bucket to Redshift cluster
```mysql
CREATE external schema orderlog_schema
FROM data catalog
database 'orderlogs'
iam_role 'arn:aws:iam::669815420608:role/RedshiftSpectrum'
region 'southeast-1';
```

Query
```mysql
SELECT description, count(*)
FROM orderlog_schema.order_logs_sundogedu
WHERE country = 'France'
AND year = '2019'
AND month = '02'
GROUP BY description;
```



---
# Redshift Quiz



1. You are working as Big Data Analyst of a data warehousing company. The company uses RedShift clusters for data analytics. For auditing and compliance purpose, you need to monitor API calls to RedShift instance and also provide secured data. Which of the following services helps in this regard ?
    - CloudTrail logs [CORRECT]
    - CloudWatch logs 
    - Redshift Spectrum
    - AmazonMQ



2. You are working as a Big Data analyst of a Financial enterprise which has a large data set that needs to have columnar storage to reduce disk IO. It is also required that the data should be queried fast so as to generate reports. Which of the following service is best suited for this scenario?
    - Redshift



3. You are working for a data warehouse company that uses Amazon RedShift cluster. It is required that VPC flow logs is used to monitor all COPY and UNLOAD traffic of the cluster that moves in and out of the VPC. Which of the following helps you in this regard ?
    - By enabling Enhanced VPC routing  on Redshift cluster



4. You are working for a data warehousing company that has large datasets (20TB of structured data and 20TB of unstructured data). They are planning to host this data in AWS with unstructured data storage on S3. At first they are planning to migrate the data to AWS and use it for basic analytics and are not worried about performance. Which of the following options fulfills their requirement?
    - node type ds2.xlarge
    - Since they are not worried about performance, storage (ds) is more important than computing power (dc,) and expensive 8xlarge instances aren't necessary.



5. Which of the following services allows you to directly run SQL queries against exabytes of unstructured data in Amazon S3?
    - Redhsift Spectrum



---
# Amazon Relational Database System (RDS)

Hosts relational database like MySQL, PostgreSQL, ... Not meant for big data, can migrate to Redshift

ACID compliance
- Atomicity
    - Ensures entire transaction as a whole is successfully executed or if part of a transaction fails then entire transaction is invalidated
- Consistency
    - Ensures that data written into database as part of the transaction must adhere to all defined rules and restrictions including constraints cascades and triggers
- Isolation
    - Ensures that each transaction is independent unto itself
        - Critical in achieving concurrency control
- Durability
    - Ensures all changes made to database is permanent once a transaction is successfully completed



---
## Resources:


