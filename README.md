### Example commands

1. run vanilla GCN on flickr, one run
```
python main.py --dataset flickr --method baseline --model2 GCN
```

2. run GPR-GNN on ogbn-arxiv, one run
```
python main.py --dataset arxiv --method baseline --model2 GPR-GNN
```

3. run Mowst*-GCN on ogbn-products, one run, and the input features for the gating module only contain dispersion
```
python main.py --dataset product --method mowst_star --model2 GCN --original_data false
```

4. run Mowst-Sage on pokec, one run, and the input features for the gating module contain dispersion and the node self-features
```
python main.py --dataset pokec --method mowst --model2 Sage --original_data true
```

5. run Mowst-Sage on penn94 (grid search, 10 runs), and the input features for the gating module contain dispersion and the node self-features
```
python main.py --dataset penn94 --method mowst --model2 Sage --original_data true --setting ten
```
