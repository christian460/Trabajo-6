import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv('ReportePCBienes202310.csv', encoding='ISO-8859-1', delimiter=';')

features = ['TOTAL', 'TIPO_PROCEDIMIENTO', 'ENTIDAD']
numeric_features = ['TOTAL']
categorical_features = ['TIPO_PROCEDIMIENTO', 'ENTIDAD']
scaler = StandardScaler()
data['TOTAL'] = scaler.fit_transform(data[['TOTAL']])

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(data[categorical_features])
data_encoded = pd.concat([data[numeric_features], pd.DataFrame(encoded_features)], axis=1)
data_encoded.columns = data_encoded.columns.astype(str)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data_encoded['Cluster'] = kmeans.fit_predict(data_encoded)

print(data_encoded)