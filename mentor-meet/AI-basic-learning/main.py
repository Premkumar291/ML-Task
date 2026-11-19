from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print("Input shape:", X.shape)
print("Target shape:", y.shape)

print("Features:", iris.feature_names)
print("Classes:", iris.target_names)