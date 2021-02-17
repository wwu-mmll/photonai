<h1>Caching</h1>
PHOTONAI offers a specialized caching that handles partially overlapping hyperparameter configurations 
for nested cross-validation splits. This is particularly useful for reusing results from expensive computations. 

More generally, caching is useful whenever re-computation needs more time than loading data. 

It is easily enabled by adding the _cache_folder_ parameter to the <a href="../../api/base/hyperpipe">Hyperpipe</a>.

```python
pipe = Hyperpipe("...",
                 cache_folder="./cache")
```