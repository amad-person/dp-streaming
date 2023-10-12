# Differential Privacy for Streaming Data

## Problem Statement
- Supported operations on a streaming database: insertion, deletion, no op.
- Batch operations per day, i.e., release answers at the end of each day.
- Distribute the desired epsilon, delta over the following mechanisms:
  - Count number of deletions to choose which mechanism to use:
    - Run naive binary mechanism on deletions to answer linear queries with PMW.
    - Run binary mechanism with restarts to answer linear queries with PMW.
  - Count number of deletions per node.

## Design

![dp-streaming.png](dp-streaming.png)

### Query 

A query object encapsulates the desired privacy parameters, sensitivity, and the noise mechanism being used to compute a function over the dataset.

The following queries are currently supported:
1. Count Query: Return the number of records in the dataset.
1. Predicate Query: Return the number of records in the dataset that satisfy the specified predicate (e.g., number of people who are 25 years old).
1. PMW Query (TBA): Estimate the synthetic data distribution that is 'trained' using the set of queries and return answers for each query in this set.

You can define your own queries by extending the base `Query` class. Your class should contain the following methods:
1. `set_privacy_parameters()`: Update the privacy parameters (epsilon, delta) for the query. This is useful when you want to change the privacy parameters specified during query initialization. 
2. `get_true_answer()`: Return the answer of the query on the dataset, without noise added.
3. `get_private_answer()`: Return the answer of the query on the dataset, after noise has been added using a desired private mechanism (e.g., Laplace). 

### Dataset

A compatible dataset should have the following columns:
1. ID: This column contains a unique identifier for each record.
1. Insertion Time: This column contains the timestamp of when the record was inserted into the dataset. 
1. Deletion Time: This column contains the timestamp of when the record was removed from the dataset. 

### Query Engine

The query engine encapsulates how the dataset is processed as a stream. A query engine takes a Query and a Dataset as input, and returns answers according to the desired time interval (e.g., release answers each day). Each query engine defines a tree representation for the stream. This representation is in turn determined by the type of node used by the tree.   

The following query engines are currently supported: 
1. Naive Binary: Two binary streams made of NaiveNodes. One stream processes insertions and the other processes deletions.
1. Binary Restarts: A single binary stream made of RestartNodes. Each node consists of insertions and its own deletion stream to track if any of its items have been deleted. 

#### Nodes

A node is a unit of the tree that represents the dataset stream. 

The following nodes are currently supported: 
1. NaiveNode: A simple node that computes the query on the IDs added to the node on initialization.
1. RestartNode: A node that tracks how many IDs added to the node on initialization are deleted, and refreshes the node once there are enough deletions.
