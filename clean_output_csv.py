import pandas as pd

input_file = 'output_cleaned.csv'
df = pd.read_csv(input_file)

print('Label counts before cleaning:')
print(df['label'].value_counts(dropna=False))

# Keep only valid labels
valid_labels = [-1, 0, 1]
df = df[df['label'].isin(valid_labels)]
df['label'] = df['label'].astype(int)

print('\nLabel counts after cleaning:')
print(df['label'].value_counts(dropna=False))

df.to_csv(input_file, index=False)
print(f'Cleaned file saved to {input_file}') 