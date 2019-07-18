---
interact_link: content/machine-learning/aws/aws-big-data/aws-big-data-specialty-processing.ipynb
kernel_name: python3
has_widgets: false
title: 'Processing'
prev_page:
  url: /machine-learning/aws/aws-big-data/aws-big-data-specialty-collection
  title: 'Collections'
next_page:
  url: /machine-learning/aws/aws-big-data/aws-big-data-specialty-storage
  title: 'Storage'
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
# AWS Lambda
Serverless Data Processing

- A way to run code snippets in cloud
    - Serverless $\therefore$ No need to provision a bunch of EC2 servers to run the code
        - Cost savings as you're only paying for the time you process and not when the EC2 servers are idle
    - Continuous Scaling (Lambda automatically scales out hardware to as much needed depending on how much data goes into it)
    - E.g. for our order history app previously, we can remove the EC2 instance that was helping us connect the Kinesis Data Streams / Firehose using the Kinesis Agent (Producer) to our consumers, DynamoDB / S3 Bucket to store logs and use AWS Lambda instead to do this, triggered by events.
    
Use Cases:
- Real-time file processing
- Real-time stream processing
- Extract Transform Load Pipelines
- Cron replacement
- Process AWS events
    - Lambda functions can be triggered by a variety of AWS sources
        - E.g. When an object on S3 changes, it can trigger a lambda function 
        
Lambda + Kinesis:
- Your Lambda code receives an event with a batch of stream records
    - You specify a batch size when setting up the trigger (up to 10,000 records)
    - Too large a batch size can cause timeouts! (Max time before timeout is 900s)
    - Batches may also be split beyond Lambda’s payload limit (6 MB)
- Lambda will retry the batch until it succeeds or the data expires
    - This can stall the shard if you don’t handle errors properly
    - Use more shards to ensure processing isn’t totally held up by errors
- Lambda processes shard data synchronously

Promises:
- Unlimited scalability

Anti-patterns:
- Long-running applications
    - If you're performing heavy-duty processing, it's advised to port it over to an EC2 instance or chain lambda functions to work on different parts of the processing because we only have 900s per lambda function
- Dynamic websites
    - EC2 and CloudFront is a better choice for this
- Stateful applications
    - Because Lambda functions can't share any information with one another, you can store state information in DynamoDB or S3 to work around the stateless nature of lambda



---
# Order History App Final

It's time to convert our `Consumer.py` script into an AWS Lambda function. A single script running on an EC2 instance is not scalable compared to using Lambda functions that run in serverless environments that will scale automatically.

1. Create IAM roles so that lambda function can consume data from Kinesis Data Streams and write data into DynamoDB
    1. Go to IAM console
    2. Go to `Roles` and `Create Role`
    3. Choose `Lambda` as the service that will use this role
    4. `Next Permissions` and specify the policies that we want to attach to this role
        - `AmazonKinesisReadOnlyAccess`
        - `AmazonDynamoDBFullAccess`
    5. Role name: `CadabraOrders` and create role
2. Create Lambda function
    1. Go to Lambda console
    2. Click `Create Function`
    3. Select `Author from scratch` and Name: `ProcessOrders`
    4. Runtime: `Python 2.7`
    5. Choose an existing role: `CadabraOrders` and create function
    6. Create trigger to make sure that when new data is being received into our Kinesis Data Stream, it will cause out function to be invoked
        - Click on `Kinesis` on side menu and configure it
            - Scroll down and click `Add` and `Save`
        - Click `ProcessOrders` to write the code needed to actually process the orders (same thing as what the Kinesis Agent flow was helping us perform)
            - Copy and paste the code below into the IDE at the bottom of the page and `Save`
    7. Run `sudo ./LogGenerator.py 10` and check DynamoDB table to see if it worked.

### lambda-function
```python
import base64
import json
import boto3
import decimal

def lambda_handler(event, context):
    item = None
    dynamo_db = boto3.resource('dynamodb')
    
    # Specify where we want Lambda to send the data to
    table = dynamo_db.Table('CadabraOrders')
    
    # We have to decode the kinesis data
    decoded_record_data = [base64.b64decode(record['kinesis']['data']) for record in event['Records']]
    
    deserialized_data = [json.loads(decoded_record) for decoded_record in decoded_record_data]

    with table.batch_writer() as batch_writer:
        for item in deserialized_data:
            invoice = item['InvoiceNo']
            customer = int(item['Customer'])
            orderDate = item['InvoiceDate']
            quantity = item['Quantity']
            description = item['Description']
            unitPrice = item['UnitPrice']
            country = item['Country'].rstrip()
            stockCode = item['StockCode']
            
            # Construct a unique sort key for this line item
            orderID = invoice + "-" + stockCode

            batch_writer.put_item(                        
                Item = {
                                'CustomerID': decimal.Decimal(customer),
                                'OrderID': orderID,
                                'OrderDate': orderDate,
                                'Quantity': decimal.Decimal(quantity),
                                'UnitPrice': decimal.Decimal(unitPrice),
                                'Description': description,
                                'Country': country
                        }
            )

```



---
# AWS Lambda Quiz



1. You are going to be working with objects arriving in S3. Once they arrive you want to use AWS Lambda as a part of an AWS Data Pipeline to process and transform the data. How can you easily configure Lambda to know the data has arrived in a bucket?
    - Configure S3 bucket notifications to Lambda
    - Lambda functions are generally invoked by some sort of trigger. S3 has the ability to trigger a Lambda function whenever a new object appears in a bucket.



2. You are going to analyze the data coming in an Amazon Kinesis stream. You are going to use Lambda to process these records. What is a prerequisite when it comes to defining Lambda to access Kinesis stream records ?
    - Lambda must be in the same account as the service triggering it, in addition to having an IAM policy granting it access.



3. How can you make sure your Lambda functions have access to the other resources you are using in your big data architecture like S3, Redshift, etc.?
    - Having the proper IAM roles
    - IAM roles define the access a Lambda function has to the services it communicates with.



4. You are creating a Lambda - Kinesis stream environment in which Lambda is to check for the records in the stream and do some processing in its Lambda function. How does Lambda know there has been changes / updates to the Kinesis stream ?
    - Lambda polls your Kinesis streams for new activity, not that Kinesis streams push to Lambda



5. When using an Amazon Redshift database loader, how does Lambda keep track of files arriving in S3 to be processed and sent to Redshift ?
    - In a DynamoDB table



---
# AWS Glue

Defines Table definitions and perform ETL pipeline on underlying data lake and provide structure to unstructured data.

What is Glue?
- Central metadata repo for data lake in S3
- Custom ETL jobs that are using Apache Spark behind the scenes

AWS Glue Crawler
- Crawls S3 and helps to detect schema in our .csv and .tsv files in S3 and start populating the Glue data catalogue with the inferred table definitions
    - E.g. Column names, Column Types, ... These are stored in the data catalogue and vented to other services like Redshift, Athena, and EMR

Glue and S3 Partitions
- It is important to structure the data in S3 properly so that it can be easily queried for your use case
    - E.g. If we're going to be querying data first by device type, it'll make sense for device type / id to be at the higher level of our directory, but if we'll be expecting to query by date, it's better to have yyyy/mm/dd/device-type/id to be the file structure instead.
    
Glue + Hive
- Use AWS Glue data catalogue as metadata store for hive or import hive meta store into glue
    - Glue data catalogue can provide meta data to hive on ElasticMapReduce
    
Glue ETL
- Automatically generate python / scala code after we definied the transformations on the data
- Security
    - Server-side (at rest)
    - SSL (in transit)
- Can trigger ETL job with events
- Can provision additional Data processing units (DPU) to handle larger Spark Jobs in ETL
- Errors are reported to CloudWatch and we can integrate SNS to notify those errors

Glue Costs
- Billed by minute of when crawler is used and ETL jobs
- Developing code on ETL jobs will also be billed by the minute

Glue Anti-Patterns
- Glue is batch oriented and at minimum can only deal with 5 min intervals, hence should not be used for streaming data
    - Kinesis should be used if ETL needs to be performed on streaming data, then store it in S3 / Redshift, then trigger Glue ETL to continue transforming it if need be
- If ETL needs to be performed in other engines like Hive / Pig, EMR might be a better choice to perform the ETL than Glue ETL
- Glue does not support NoSQL databases like DynamoDB
    - The entire purpose of Glue is to figure out structured / rigid schema databases like SQL databases



---
# AWS Glue Quiz



1. You want to load data from a MySQL server installed in an EC2 t2.micro instance to be processed by AWS Glue. What applies the best here?
    - Instance should be in your Virtual Private Cloud 
    - Although we didn't really discuss access controls, you could arrive at this answer through process of elimination. You'll find yourself doing that on the exam a lot. This isn't really a Glue specific question; it's more about how to connect an AWS service such as Glue to EC2.
    - MySQL does not have in-built Glue support



2. What is the simplest way to make sure the metadata under Glue Data Catalog is always up-to-date and in-sync with the underlying data without your intervention each time?
    - Schedule Crawlers to run periodically



3. Which programming languages can be used to write ETL code for AWS Glue?
    - Python and Scala



4. Can you run existing ETL jobs with AWS Glue?
    - You can run your existing Scala or Python code on AWS Glue. Simply upload the code to Amazon S3 and create one or more jobs that use that code. You can reuse the same code across multiple jobs by pointing them to the same code location on Amazon S3.



5. How can you be notified of the execution of AWS Glue jobs?
    - AWS Glue outputs its progress into CloudWatch, which in turn may be integrated with the Simple Notification Service.



---
# AWS EMR

A collection of EC2 instances:
- Master Node
    - Manages the cluster, monitors the health of other nodes in the cluster
- Core node
    - Hosts HDFS data and run tasks
    - Adding core nodes might run the risk of losing partial data because data is stored on core nodes
- Task node
    - Only Runs task, does not host data
    - Can just add these as needed as the traffic into cluster increases
    - Good use of spot instance (temporary EC2 instance) for on demand
    
Transient Cluster
- Configure a cluster to be automatically terminated after it completes some tasks 
- Very inexpensive because you're only paying for capacity you need

Long-Running Cluster
- E.g. using the cluster as a data warehouse and preiodic processing is done on a large dataset

EMR/AWS Integration
- EMR uses EC2 isnatnces as the nodes of the cluster
- Amazon manages the nodes instead of you managing your own Hadoop nodes
- Can run Cluster within a virtual network for security purposes 
- Allows S3 to be source of input and output
- Amazon CloudWatch to monitor cluster performance and configure alarms
- CloudTrail to create an audit log of all the requests made to the service
- AWS IAM to configure permissions
- AWS CloudTrail to audit requests made to the service
- AWS Data Pipeline to schedule and start your clusters

Hadoop Distributed File System (HDFS)
- Each node stores blocks of data that's 128 MB
    - So if storing a big file on HDFS, it'll be split up into 128MB chunks
- If we terminate the cluster, data is gone
- Useful for caching intermediate results during MapReduce processing or for workloads that have very significant random input/output
- Only use this option if we never have to shut down the cluster
    - If we know that we're going to terminate the cluster at any point in time, better to store data elsewhere like EMRFS
    
ElasticMapReduce File System (EMRFS)
- Data is instead stored in S3, so even if you terminate, the data lives on
- We can use both HDFS and S3 as the file system in the cluster so that you can store input / output data on S3 while intermediate results on HDFS
- EMRFS Consistent View
    - When a node is trying to write to the same place where another node is trying to access data from S3, we have a consistency problem
        - This isnt an issue with HDFS because data tends to be processed on the same node it's stored in
    - When EMRFS Consistent View is enabled, EMR will use a DyanmoDB database to store object metadata and track consistency with S3 
    
Local File System
- When you have locally connected disks on the EC2 nodes and those will be pre configured with an instance store
- Only useful for storing temporary data that's continually changing like buffers / caches / scratch data

Elastic Block Store (EBS) for HDFS
- Allows us to runb EMR clusters on EBS-only EC2 instances like M4 / C4
- EMR will delete these volumes once cluster is terminated
- Cannot attach EBS to a a running cluster, so you must add EBS volume when launching a cluster
- Manually detaching EBS volume will cause a failure and both instance storage and volume storage will be replaced

EMR promises
- Charges by the hour + EC2 charges
- Can use transient cluster if we dont need data to persist for long
- Can add / remove task nodes on the fly
- Can resize core nodes

Hadoop
- Hadoop Common
    1. MapReduce
        - Framework to write applications to process vast amounts of data in parallel
        - Map functions to sets of key value pairs (intermediate results)
        - Reduce functions to combine intermediate results
    2. Yet Another Resource Negotiator (YARN)
        - Manages what gets run on which node
    3. HDFS
        - Distributes data cross instances in the cluster and creates multiple copies of the data
- Instead of pure MapReduce, we can use Apache Spark

## Pre-installed

### Apache Spark
- Uses memory caching so does a lot of work in-memory instead of on disk
- Uses DAGs to optimize query execution for very fast analytic queries against data of any size
- Use cases:
    - Stream processing with Spark Streaming
        - Process data collected from Kinesis 
        - Process data from Apache Kafka (A 3rd party Producer that can produce to Kinesis Data STreams)
    - Perform Streaming analytics and send data to HDFS or S3
    - Mllib for ML for massive data
    - Spark SQL Database
        - Use SQL or HIveQL 
        - Not meant for OLTP (online transaction processing)
        - Not meant for Batch processing because Spark jobs take time to complete because of the distribution of work, not meant for Real-time usage
        - Used for analytical applications / large tasks on a regular schedule
- Driver program (Your main code) contains a Spark context object that will communicate with the cluster manager of choice (YARN) that would allocate the resources needed for the driver program.
- Once cluster manager has decided on how to distribute, Spark will acquire the executors on nodes within the cluster
    - Executors are processes that run computations and store the data for your application
    - Application code is sent to each executor
    - Spark context sends tasks to executors to run
- Features:
    - Spark Core
        - responsible for memory management, fault recovery, scheduling, distributing and monitoring jobs, and interacting with storage systems
        - **How does Spark store it's data?**
            - Spark Resilient Distributed Dataset
            - On a higher level, we are using Spark SQL
    - Spark SQL 
        - distributed query engine that provides low latency queries of up to 100 times faster than MapReduce.
            - Contains Cost-based optimizer, Columnar storage (Can read data by columns, skipping unecessary searching), Cogeneration for fast queries
            - Supports various data sources coming in
                - JDBC, ODBC, JSON, HDFS, Hive, Orc, Parquet
    - Spark Streaming
        - Ingests data in mini batches and enables analytics on that data by storing the data into a "data set", a table that keeps growing and we can query the table according to window of time 
        - `val inputDF = spark.readStream.json("s3://logs")`
        - `inputDF.groupBy($"action", window($"time", "1 hour")).count()`
        - `.writeStream.format("jdbc").start("jdbc:mysql//...")`
            - Surpports data from 
                - Kafka, Flume, HDFS, Zero MQ, Kinesis
            - Kinesis + Spark Streaming
                - Spark consumes from Kinesis Datastream and creates a Spark Dataset fropm that and now we can process that dataset across EMR
    - MLLib
        - ML algorithms
            - Supports reading data from HDFS, HBase, any Hadoop Data source, S3 on EMR
    - GraphX
        - Distributed graph processing framework
            - Provides ETL capabilities, Exploratory analysis, Iterative graph computation
            - Social network analysis

Spark + Redshift
- Redshift is a massively distributed data warehouse
- `spark-redshift` package converts data from redshift to Spark datasets 
- Perform ETL on this using EMR cluster to distribute workload
- E.g. A bunch of airline flight data is stored in a data lake in S3
    - Use Redshift spectrum on top of that data to get an SQL interface 
    - Use `spark-redshift` to perform ETL on EMR with the Spark dataset created
    - Can take that processed data to load into Redshift to further process
    
### Apache Hive
- Hive is a built on top of EMR cluster so that we use HiveQL (SQL-like)
    - Used for OLAP (Online Analytical Processing) applications
    - Shouldnt be used for OLTP, shouldnt be writing a web service that hits HIve continuously hundreds of times per seciond
- Tez can take the place of MapReduce as it is very similar to Spark in that it
    - Uses memory caching so does a lot of work in-memory instead of on disk
    - Uses DAGs to optimize query execution for very fast analytic queries against data of any size
- Hive metastore
    - Similar to AWS Glue Data catalogue
    
### Pig
- Alternative to writing MapReduce code but not quite SQL
    - Highly distributed scripting language that looks like SQL
    
### HBase
- Non-relational NoSQL database
- Very fast because it operates in-memory, and not doing a bunch of disk seeks
- Integrates with hive to issue SQL style commands 
- Performs the same role as DynamoDB
    - Use DynamoDB when you'll be using alot AWS Services 
    
### Presto
- Performing interactive queries at the Petabyte scale across a wide variety of data sources
- Uses SQL syntax, optimized for OLAP applications 
- Amazon Athena uses this behind the scenes, making Athena a serverless version of Presto
- Exposes JDBC, command line, Tableau interfaces 
- Presto Connectors
    - HDFS
    - S3
    - Cassandra
    - MongoDB
    - HBase
    - SQL
    - Redshift
    - Teradata
- Not good for OLTP or batch processing

### Apache Zeppelin
- Like Jupyter Notebook

### Hadoop User Experience (HUE)
- UI for managing your clusters

### Splunk
- Similar to Hue, gathering data all the time about how your cluster is actually performing
- Just on operational tool

### Flume
- Streams data to your cluster like Kinesis or Kafka
- Purpose is for log data coming in from a big fleet of web servers
    - Web servers will provide events to flume source
    - Event is stored in one or more channels
    - Channels keeps event until it's consumed by a flume sink, then removes event from channel and puts in an external repo like HDFS / EMR cluster
    
### MXNet
- Tensorflow alternative

### S3DistCP
- Tool for copying large amounts of data from
    - S3 to HDFS
    - HDFS to S3
- Uses MapReduce to copy large number of objects in parallel using entire cluster

### Ganglia
- Monitoring tool similar to CloudWatch

### Mahout
- ML library like MLlib

### Accumulo
- Another NoSQL database but more for security

### Derby
- Anither relational database in Java

### Sqoop
- Relational database connector
- Imports data from external sources in a scalable manner
- Parallelizes copying of external relational database into cluster

### HCatalog
- Meta layer on top of Hive metastore

### Kinesis Connector
- Access Kinesis streams from any script you might be writing

### Tachyon
- Speeds up Apache Spark 

### Ranger
- Data security manager

## EMR Security
- IAM policies
    - Tagging also to control access on a cluster by cluster basis
- Kerberos
    - Providing strong authentication using secret key cryptography
    - Network authentication protocol that ensures passwords or other credentials arent sent over the network in an unencrypted format
- SSH
    - Kerberos or EC2 key pairs can be used to authenticate clients for SSH and as a means of encrypting data in transit
- IAM roles

## Choosing instance types for EMR Cluster
- Master node:
    - m4.large if < 50 nodes, m4.xlarge if > 50 nodes
- Core & task nodes:
    - m4.large is usually good
    - If cluster waits a lot on external dependencies (i.e. a web crawler), t2.medium
    - Improved performance: m4.xlarge
    - Computation-intensive applications: high CPU instances
    - Database, memory-caching applications: high memory instances
    - Network / CPU-intensive (NLP, ML) – cluster computer instances
- Spot instances
    - Good choice for task nodes
    - Only use on core & master if you’re testing or very cost-sensitive; you’re risking partial data loss



---
# Requirement 2: Product Recommendations

$$
\text{Kinesis Firehose} \rightarrow \text{S3} \rightarrow \text{EMR}
$$

1. Setup EMR Cluster
    1. Select EMR on Management Console
    2. `Create Cluster` and Cluster name: `CadabraRecs`
    3. Choose `Spark` Application because we want to use MLlib to generate recommendations
    4. Choose instance type `m5.xlarge` for your EC2 instances to act as the EMR nodes (Maybe a `c5` might be more appropriate as we're performing machine learning which is CPU intensive, if we're doing something deep learning related, GPU optimized instances might be a better choice)
    5. Choose Number of instances: `3` (1 master and 2 core nodes)
    6. Choose a EC2 Key pair like the one we've created in the previous project and `Create Cluster`
    7. Connect to master node from Terminal so that we can run our Spark script
        1. Scroll down and Click on `Security Groups for Master`
        2. Select Group Name: `ElasticMapReduce-master` and click `Inbound` - Notice that port 22 isn't here so we can't connect through Terminal using SSH
        3. Click `Edit` and `Add Rule`
            - Input `Custom TCP Rule`, `22`, `My IP`
            - `Save`
        4. Go back and click on `Master public DNS`
            - Click `SSH` and we'll have instructions on how to connect
    8. Use one of the samples that come with Spark that recommends using Alternating Least Squares
        1. Make copy of home directory in case we mess up
            - `cp /user/lib/spark/examples/src/main/python/ml/als_example.py ./`
        2. Check if the script will run
            - Copy movie lens data from our local library in EC2 master node into HDFS
                - `hadoop fs -mkdir -p /user/hadoop/data/mllib/als`
                - `-copyFromLocal /user/lib/spark/data/mllib/als/sample_movielens_ratings.txt /user/hadoop/data/mllib/als/sample_movielens_ratings.txt`
                - Add `spark.sparkContext.setLogLevel("ERROR")` into `als_example.py` to only log messages that are on the ERROR level
                - `spark-submit als_example.py`
        3. Configure `als_example.py` to read data from our S3 Bucket instead of using movie lens dataset
            - `spark-submit als_example.py`
        
### Edited als_example.py
```python
from __future__ import print_function

import sys
if sys.version >= '3':
    long = int

# Importing pyspark which is a python spark driver script
from pyspark.sql import SparkSession

# $example on$
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
# $example off$

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()
    
    # ADDED: Only display logs that are ERROR
    spark.sparkContext.setLogLevel("ERROR")

    # $example on$
    # This is not a local file path, Spark will look for this
    # in the HDFS on this cluster
    
    # REMOVED:
    # lines = spark.read.text("data/mllib/als/sample_movielens_ratings.txt").rdd
    # parts = lines.map(lambda row: row.value.split("::"))
    # ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=long(p[3])))
                                         
    # ADDED:
    lines = spark.read.text("s3://orderlogs-sundogsoft/2019/02/11/16/*").rdd # Location of the file in S3
    parts = lines.map(lambda row: row.value.split(','))
    #Filter out postage, shipping, bank charges, discounts, commissions
    productsOnly = parts.filter(lambda p: p[1][0:5].isdigit())
    #Filter out empty customer ID's
    cleanData = productsOnly.filter(lambda p: p[6].isdigit())
    ratingsRDD = cleanData.map(lambda p: Row(customerId=int(p[6]), \
        itemId=int(p[1][0:5]), rating=1.0))
   
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    
    # REMOVED:
    # als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
              
    # ADDED:
    als = ALS(maxIter=5, regParam=0.01, userCol="customerId", itemCol="itemId", ratingCol="rating",
              coldStartStrategy="drop")
    
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)

    # Generate top 10 movie recommendations for a specified set of users
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of movies
    movies = ratings.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    # $example off$
    userRecs.show()
    movieRecs.show()
    userSubsetRecs.show()
    movieSubSetRecs.show()

    spark.stop()
```



---
# EMR and Hadoop Quiz



1. Of the following tools with Amazon EMR, which one is used for querying multiple data stores at once?
    - Presto



2. Which one of the following statements is NOT TRUE regarding EMR Notebooks?
    - EMR notebooks stopped if idle for extended time [TRUE]
    - EMR notebooks currently do not integrate with repos for version control [TRUE]
    - EMR Notebooks can be opened without logging into AWS Management console [FALSE] 
        - To create or open a notebook and run queries on your EMR cluster you need to log into the AWS Management Console.
    - You cannot attach your notebook to a Kerberos enabled EMR cluster [TRUE]



3. How can you get a history of all EMR API calls made on your account for security or compliance auditing?
    - AWS CloudTrail integration is one of the ways in which EMR integrates with AWS.



4. When you delete your EMR cluster, what happens to the EBS volumes?
    - EMR will delete volume once cluster is terminated
    - If you don't want the data on your cluster to be ephemeral, be sure to store or copy it in S3.



5. Which one of the following statements is NOT TRUE regarding Apache Pig?
    - Pig supports interactive and batch cluster types [TRUE]
    - Pig is operated by a SQL-like language called Pig latin [TRUE]
    - When used with EMR, Pig allows accessing multiple filesystems [TRUE]
    - Pig supports access through JDBC [FALSE]



---
# Amazon Sagemaker
- Develop models on Jupyter notebook, do not have to worry about provisioning capacity for ML jobs
- 3 Modules
    - Build
        - Hosted environment for working with data
        - Here is where youll work on jupyter notebooks 
        - Notebooks are pre-loaded with CUDA and CUDANN drivers for deep learning
        - Download Docker container to local environment to develop models for Sagemaker
    - Train
        - One click model training and tuning at high scale low cost
        - Sagemaker Search to quickly find and evaluate most relevant model training runs 
    - Deploy
        - Provides a managed environment to host and test models to make predictions securely with low latency
        - Batch transform mode to run predictions on small batch data
- Built-in CloudWatch monitoring and logging
- Sagemaker Neo 
    - Allows ML models to be trained once and run anywhere in the cloud and push them to the edge nodes to make the prediction fast

Security:
- Code is stored in ML Storage volumes
    - Controlled by security groups
    - optionally encrypted at rest
- CloudWatch monitors Sagemaker's metrics in near real time
- CloudTrail records history of API calls



---
# Sagemaker Quiz



1. What limit, if any, is there to the size of your training dataset in Amazon Machine Learning by default?
    - 100GB



2. The audit team of an organization needs a history of Amazon SageMaker API calls made on their account for security analysis and operational troubleshooting purposes. Which of the following service helps in this regard?
    - SageMaker outputs its results to both CloudTrail and CloudWatch, but CloudTrail is specifically designed for auditing purposes.



3. Is there a limit to the size of the dataset that you can use for training models with Amazon SageMaker? If so, what is the limit?
    - There are no fixed limits to the size of the dataset you can use for training models with Amazon SageMaker.



4. Which of the following is a new Amazon SageMaker capability that enables machine learning models to train once and run anywhere in the cloud and at the edge?
    - Sagemaker Neo



5. A Python developer is planning to develop a machine learning model to predict real estate prices using a Jupyter notebook and train and deploy this model in a high available and scalable manner. The developer wishes to avoid worrying about provisioning sufficient capacity for this model. Which of the following services is best suited for this?
    - Sagemaker is fully managed



---
# AWS Data Pipeline
- Schedule tasks to complete regularly like transferring data from EC2 to S3 and end of week transfer that data onto an EMR cluster so that we can run analysis on it...
- Destinations:
    - S3
    - RDS
    - DynamoDB
    - Redshift
    - EMR
- Manages task dependencies and retries and notifies on failures
- cross-region pipelines
- Precondition checks
    - DynamoDB Data Exists check
            - Checks whether an entire table for S3 exists
    - Shell command precondition to run a script that checks for whatever you want 
- Use On-premise data

Data Pipeline Activities
- EMR
- Hive
- Copy
- SQL
- Scripts



---
## Resources:


