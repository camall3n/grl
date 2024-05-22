import seaborn as sns


hex = [
    '#ff96b6',
    '#df5b5d',
    '#DD8453',
    '#f8de7c',
    '#3FC57F',
    '#48dbe5',
    '#3180df',
    '#9d79cf',
    '#886a2c',
    '#ffffff',
    '#d5d5d5',
    '#666666',
    '#000000',
]
p = sns.color_palette(hex, as_cmap=False)
p

for i in range(len(p)):
    print(f'{hex[i]}: {p[i]}\n')
