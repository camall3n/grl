import seaborn as sns


hex = [
    '#3180df',
    '#df5b5d',
    '#9d79cf',
    '#f8de7c',
    '#d5d5d5',
    '#ffffff',
    '#000000',
    '#48dbe5',
    '#ff96b6',
    '#886a2c',
]

p = sns.color_palette(hex, as_cmap=False)

for i in range(len(p)):
    print(f'{hex[i]}: {p[i]}\n')
