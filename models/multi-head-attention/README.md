# Multi-Head Attention

The module splits it's Query, Key, and Value parameters N ways, and passes the splits independently to a seperate attention head.
In the end, the individual attention values are added back together

## Steps:
1. Split input into multiple heads.
2. Each head computes its own attention.
3. Concatenate the results.
4. Pass through a final linear layer

