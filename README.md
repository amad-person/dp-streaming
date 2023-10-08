# Differential Privacy for Streaming Data

## Problem Statement
- Supported operations on a streaming database: insertion, deletion, no op.
- Batch operations per day, i.e., release answers at the end of each day.
- Distribute the desired epsilon, delta over the following mechanisms:
  - Count number of deletions to choose which mechanism to use:
    - Run naive binary mechanism on deletions to answer linear queries with PMW.
    - Run binary mechanism with restarts to answer linear queries with PMW.
  - Count number of deletions per node.

## Pseudocode

```
Inputs to Engine: Queries, Epsilon, Delta, DeletionFreqThreshold
Initialize Engine:
    NaiveBinaryInsertionsMap // key: day, value: NaivePMWNode
    NaiveBinaryDeletionsMap // key: day, value: NaivePMWNode
    BinaryRestartsDeletionsMap // key: day, value: RestartPMWNode
On each day D:
    // Update data structures
    Update NaiveBinaryInsertions using insertions in the day. 
        Compute epsilon and delta for the node.
        Add all insertions to a new NaivePMWNode with epsilon and delta. 
            Combine any previous nodes with this one (e.g. making a parent node with two children). 
        Update day -> Node mapping to NaiveBinaryInsertionsMap for querying this node on future days.
    Update NaiveBinaryDeletions using deletions in the day. 
        Compute epsilon and delta for the node.
        Add all insertions to a new NaivePMWNode with epsilon and delta. 
            Combine any previous nodes with this one (e.g. making a parent node with two children). 
        Update day -> Node mapping to NaiveBinaryDeletionsMap for querying this node on future days.
    Update BinaryRestartsDeletions using insertions in the day.
        Compute epsilon and delta for the node.
        Add all insertions to a new RestartPMWNode with epsilon and delta.
            Combine any previous nodes with this one (e.g. making a parent node with two children). 
        Update day -> Node mapping to NaiveBinaryDeletionsMap for querying this node on future days.
    Update BinaryRestartsDeletions using deletions in the day.
        Propagate deletions to all nodes of BinaryRestartsDeletions.
    
    // Release answer
    Count the number of deletions to get CurrentNumDeletions.
    If CurrentNumDeletions > DeletionFreqThreshold:
        Release answer using NaiveBinaryInsertions and NaiveBinaryDeletions.
            Select the necessary nodes for this.
            Combine answers from each node and release.
    Else:
        Release answer using BinaryRestartsDeletions.
            Select the necessary nodes for this.
            Combine answers from each node and release.
        
Inside a NaivePMWNode:
    Initialization: 
        Run PMW on items and store.
    Return answer.

Inside a RestartPMWNode:
    Initialization: 
        Run PMW on items and store. 
        Start a NaiveBinaryDeletions to answer PMW on deletions. 
        Start a NaiveBinaryDeletionsCounter to count number of deletions. 
    If number of deletions > (number of items) / 2:
        Remove deleted items.
        Run PMW on items and store.
        Restart NaiveBinaryDeletions.
        Restart NaiveBinaryDeletionsCounter.
        Release PMW on items. 
    Else:
        Update NaiveBinaryDeletions.
        Update NaiveBinaryDeletionsCounter.
        Release answer = PMW - NaiveBinaryDeletions.
```

    