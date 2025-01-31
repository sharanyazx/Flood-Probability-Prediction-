import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
import joblib

# Load the data
filenames = ['Godavari','Krishna' ,'Cauvery','Mahanadi','Son'] # Change this to your desired river name
for filename in filenames:
    data1 = pd.read_excel("Dataset/" + filename + ".xlsx")

    # Fill missing values with mean
    data1.fillna(data1.mean(), inplace=True)

    # Preprocess target variable
    y = (data1['Flood'] >= 0.1).astype(int)  # Convert to binary

    # Extract date features
    data1['Date'] = pd.to_datetime(data1['Date'])
    data1['Year'] = data1['Date'].dt.year
    data1['Month'] = data1['Date'].dt.month
    data1['Day'] = data1['Date'].dt.day

    # Drop unnecessary columns
    data1.drop(['Date', 'Flood', 'Year', 'Month', 'Day','flood runoff'], axis=1, inplace=True)
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(data1, y, train_size=0.8, random_state=42)
    print('\n',x_train.head(),"\n",y_train.head)

    # Model Training
    Rf = RandomForestClassifier(n_estimators=200,max_depth=20,warm_start=True)
    Rf.fit(x_train, y_train)
    joblib.dump(Rf,filename+'_trained_model.pkl')

    # Model evaluation
    print("\n" + filename + ".xlsx dataset evaluation")
    print("1. Train data accuracy =", Rf.score(x_train, y_train))
    print("2. Test data accuracy =", Rf.score(x_test, y_test))

    y_predict = Rf.predict(x_test)
    print("3. Classification report:\n", classification_report(y_test, y_predict))
    mae = mean_absolute_error(y_test, y_predict)
    print("4. Mean Absolute Error =", mae)
    print("5. Confusion Matrix:\n", confusion_matrix(y_test, y_predict))

#    import shap
#    import matplotlib.pyplot as plt
#    from sklearn.linear_model import LogisticRegression
#    Lr=LogisticRegression(max_iter=1000)
#    Lr.fit(x_train,y_train)
#    # Explainable AI (XAI) using SHAP
#    shap.initjs()
#
#    # Create the explainer
#    explainer = shap.LinearExplainer(Lr, x_train)
#    shap_values = explainer.shap_values(x_test)
#
#    # Summary plots
#    shap.summary_plot(shap_values, x_test, show=False, feature_names=x_test.columns, class_names=['Low', 'High'])
#    plt.savefig("C:/my/figures/"+filename+"_dot_plot.png")
#    shap.summary_plot(shap_values, x_test, plot_type='bar', show=True, feature_names=x_test.columns, class_names=['Low', 'High'])
#    plt.savefig("C:/my/figures/"+filename+"_bar_plot.png")
#
# Save the predictions
#Total_pred = pd.DataFrame({filename: y_predict})
#Total_pred.to_csv("C:/my/pyprograms/zzz/Output.csv")
