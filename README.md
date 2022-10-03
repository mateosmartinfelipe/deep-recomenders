#  WIP
##  Neural collaborative Filter in pytorch
##### https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF


### application schema on mermaid Data transformation
```mermaid
flowchart
    A(web/url/data) --> B{Existing datasets}
    B --> |No| C(Download zip)
    C -->  E(unzip to local)
    E -->  G(local to df)
    B --> |YES| G
    G --> H(df to tensor)
    H --> Dataset
    Dataset --> Batch;
```
