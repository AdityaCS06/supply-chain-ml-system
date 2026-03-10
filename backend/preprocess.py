from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["warehouse"] = encoder.fit_transform(data["warehouse"])