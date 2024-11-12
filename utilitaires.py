
import random

def new_series_denormalisation(X_test, prediction):

    """
    Fonction pour dénormaliser les mesures météo

    """
    
    plot_real = np.zeros((X_test.shape[0], X_test.shape[1]+1, X_test.shape[2]))
    plot_prediction = np.zeros((X_test.shape[0], X_test.shape[1]+1, X_test.shape[2]))

    plot_real[:, :-1, :] = X_test #on recopie tout sauf le dernier element
    plot_real[:, -1, :] = y_test

    for i in range(plot_real.shape[0]):  # Parcours des 8235 séries
        plot_real[i, :, 2] = plot_real[i, :, 2] * ecart_type[2] + moyenne[2]

    plot_prediction[:, :-1, :] = X_test #on recopie tout sauf le dernier element
    plot_prediction[:, -1, :] = prediction

    for i in range(plot_prediction.shape[0]):  # Parcours des 8235 séries
        plot_prediction[i, :, 2] = plot_prediction[i, :, 2] * ecart_type[2] + moyenne[2]

    return plot_real, plot_prediction



def calcule_diff_moyenne(plot_real, plot_prediction):

    """
    Fonction pour calculer la différence moyenne entre les prédictions et les températures réelles

    """

    vrai_temp = plot_real[:, -1, :]
    pred_temp = plot_prediction[:, -1, :]
    ecart_moyen = (abs(vrai_temp - pred_temp)).mean()
    return ecart_moyen




def plot_and_results(plot_real, plot_prediction):

    """
    Fonction pour plot une série de 20 mesures de température, la 21 eme réelle et la 21eme prédiction

    """

    
    rn = random.randint(0, plot_real.shape[0])

    plt.figure(figsize=(10,6))
    plt.plot(plot_real[rn, :, 2], color='blue', label='Actual Temperature')
    plt.plot(plot_prediction[rn, :, 2], color='red', linestyle='--', label='Predicted Temperature')

    plt.title('Temperature Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

    print(f"la température réelle est {plot_real[rn, 20, 2]} la température prédite est {plot_prediction[rn, 20, 2]} l'écart à la prédiction est {plot_real[rn, 20, 2] - plot_prediction[rn, 20, 2]} ")


    print(plot_real[rn, 20, 2])
    print(plot_prediction[rn, 20, 2])