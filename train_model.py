## Entrenamiento del modelo

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
#Predicción en set de prueba
y_pred = lr.predict(X_test)
# Calculo de metricas
accuracy = accuracy_score(y_test, y_pred)