import pandas as pd
import numpy as np

# Dataset
data = [
    ['Technical', 'Senior', 'excellent', 'good', 'urban', 'yes'],
    ['Technical', 'Junior', 'excellent', 'good', 'urban', 'yes'],
    ['Non-Technical', 'Junior', 'average', 'poor', 'rural', 'no'],
    ['Technical', 'Senior', 'average', 'good', 'rural', 'no'],
    ['Technical', 'Senior', 'excellent', 'good', 'rural', 'yes']
]

columns = ['Role', 'Experience', 'Performance', 'InternetQuality', 'WorkLocation', 'Output']
df = pd.DataFrame(data, columns=columns)

X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])


def candidate_elimination(X, y):
    n = X.shape[1]

    # Initialize S and G
    S = ['?'] * n
    G = [['?'] * n]

    for i in range(len(X)):
        if y[i] == 'yes':   # Positive example
            for j in range(n):
                if S[j] == '?':
                    S[j] = X[i][j]
                elif S[j] != X[i][j]:
                    S[j] = '?'

            # Remove hypotheses from G inconsistent with positive example
            G = [g for g in G if all(g[j] == '?' or g[j] == X[i][j] for j in range(n))]

        else:   # Negative example
            new_G = []
            for g in G:
                if all(g[j] == '?' or g[j] == X[i][j] for j in range(n)):
                    for j in range(n):
                        if S[j] != '?' and S[j] != '?' and S[j] != X[i][j]:
                            new_h = g.copy()
                            new_h[j] = S[j]
                            if new_h not in new_G:
                                new_G.append(new_h)
                else:
                    new_G.append(g)

            G = new_G

        print(f"\nAfter instance {i+1}:")
        print("S =", S)
        print("G =", G)

    return S, G


final_S, final_G = candidate_elimination(X, y)

print("\nFinal Specific Hypothesis:", final_S)
print("Final General Hypothesis:", final_G)
