# Grounded language Learning System

General system framework for learning word-as-classifer groundings

### Prerequisites

- python 2.7 
- pandas
- genism 

## Running the tests


#### Preprocessing language input

```
python2 preprocess_descriptions.py <language name> <language> <"stop", "lemm", or "stemm">
```

#### Learning

```
python2 cLL-ML.py --resDir <folder for result output> --cat <category of learning: rgb, shape, object, all> --pre <formated language conf file> --cutoff <threshold for negative example selection> --seed <seed for random selection> --visfeat <location of visual feature folder hierarchy>  --listof <list of instances or images conf file> --negexmpl <optional: import negative examples previously computed to save time> 
```

#### Testing / Validation 

```
python2 macro-pos5DescrNegDocVecdistractorTest.py <result folder>/NoOfDataPoints/ <category: rgb, shape, object, all> <category: rgb, shape, object, all> <formated language conf file>  

```
