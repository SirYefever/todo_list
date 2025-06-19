# Natural Language Processing and Triton Inference Server
Second module ml project. Russian text normalization challenge.  

## Project setup:
1. Create `/data` folder.
2. Create `/data/inputs` folder.
3. Create `/data/results` folder.
4. Create `/data/dictionary` folder.
5. Download [these files](https://drive.google.com/drive/folders/1Oyc_d4rBBIECMZkajMik_fNbJ5x8QA6V?usp=sharing) and extract ru_train.csv and ru_test_2.csv to `/data/inputs`, put .onnx files to `/model_repository/text_normalization/1`

## Running the project:  
### To train:  
`python .\src\tsu_nlp\model\model.py train dictionary`  
### To normalize:  
`python .\src\tsu_nlp\model\model.py normalize neural` - extremely slow, full t5 normalization
`python .\src\tsu_nlp\model\model.py normalize hybrid` - pretty slow, but precise, uses t5, when dictionary can't find a match
python ./model.py normalize hybrid` - fast and precise, uses t5, when dictionary can't find a match