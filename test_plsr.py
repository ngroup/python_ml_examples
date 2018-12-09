from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, explained_variance_score
import numpy as np

data, target = load_linnerud(return_X_y=True)
scaler_data = StandardScaler()
scaler_target = StandardScaler()



print(data.shape)
print(target.shape)


# print(data)


data_scaled = scaler_data.fit_transform(data)
target_scaled = scaler_target.fit_transform(target)


pls = PLSRegression(n_components=1, scale=False)

pls.fit(data_scaled, target_scaled)

target_pred_scaled = pls.predict(data_scaled)

target_pred = scaler_target.inverse_transform(target_pred_scaled)

# print(target)
# print(target_pred)
# print(data_scaled)

print(pls.x_scores_.shape)

print(pls.x_weights_.shape)
print(pls.y_weights_.shape)


wc = np.dot(pls.x_rotations_, np.transpose(pls.y_loadings_))
print(wc)
print(pls.coef_)

print("Check X")
print(pls.x_weights_)
print(pls.x_loadings_)


print(pls.y_loadings_)
print(pls.y_weights_)


print("TC")
tc = np.dot(pls.x_scores_, np.transpose(pls.y_loadings_))
print(tc[0])
print("Pred Y")
print(target_pred_scaled[0])

print(pls.x_scores_.shape)
# x_score_1 = np.dot(pls.x_scores_, np.diag([1, 1, 1]))
# tc = np.dot(x_score_1, np.transpose(pls.y_loadings_))
print(tc)
print(target_pred_scaled[0])
# print(tc[:, 0])
print(target_scaled)


print(mean_squared_error(target_scaled[:, 0], tc[:, 0]))
print(explained_variance_score(target_scaled[:, :], tc[:, :]))


import matplotlib.pyplot as plt

fig = plt.figure()

plt.plot(data_scaled[:, 2], target_scaled[:, 2])
plt.show()

