import pandas as pd

test_data = {
    'sentence_id': [0] * 10,
    'token_id': list(range(10)),
    'before': ['В', '2023', 'году', 'я', 'купил', 'iPhone', '14', 'за', '89990', 'рублей']
}

df = pd.DataFrame(test_data)
df.to_csv('data/inputs/ru_test_2.csv', index=False) 