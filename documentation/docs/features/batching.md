<h1>Batch Processing</h1>
PHOTONAI offers batch processing of elements. This comes in handy for working memory sensitive tasks.
An example is handling large medical data modalities, such as resampling gray matter 3D brain scan niftis. 
However, be aware that this only makes sense for algorithms that independently transform each item.

Batching is easily accessed by adding the _batch_size_ parameter to the [PipelineElement](../../api/base/pipeline_element).

```python
PipelineElement("LabelEncoder", batch_size=10)
```