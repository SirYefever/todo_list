import pandas as pd

# Try different encodings
encodings = ['utf-8', 'cp1251', 'latin1']

for encoding in encodings:
    try:
        df = pd.read_csv('data/inputs/ru_train.csv', encoding=encoding, nrows=5)
        print(f"Successfully read with encoding: {encoding}")
        print("\nDataset preview:")
        print(df)
        break
    except UnicodeDecodeError:
        print(f"Failed with encoding: {encoding}")
    except Exception as e:
        print(f"Other error with encoding {encoding}: {str(e)}") 