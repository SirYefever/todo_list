# Natural Language Processing and Triton Inference Server
*In progress...*

## Project setup:
*In progress...*

## Running the project:  
*In progress...*  
### To train:  
`python ./model.py train dictionary`  
'E:\\dev\\TSU.NLP_3\\data\\inputs\\ru_train.csv'
### To normalize:  
`python ./model.py normalize dictionary` - fastest, rather precise
`python ./model.py normalize neural` - extremely slow, full t5 normalization
`python ./model.py normalize hybrid` - fast and precise, uses t5, when dictionary can't find a match

### Datasets:
You can find initial datasets for testing/training following this path:
*In progress...*

ONNX_MODEL_PATH = os.path.join(project_root, 'model_repository', 'text_normalization', '1', 'model.onnx')
ONNX_ENCODER_PATH = os.path.join(project_root, 'model_repository', 'text_normalization', '1', 'encoder_model.onnx')