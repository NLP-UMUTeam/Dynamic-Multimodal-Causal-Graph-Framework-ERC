import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_best_model_prediction.csv")
print(df)
print(classification_report(df["Emotion"], df["prediction"], digits=6))
cm = confusion_matrix(df["Emotion"], df['prediction'])

labels = sorted(df['Emotion'].unique())
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
# Crear heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Real')

# Guardar como imagen
plt.savefig('cm_meld.pdf', format='pdf', bbox_inches='tight')    
plt.close()