import os
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
import SwarmPackagePy as spp
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from SwarmPackagePy import animation, animation3D
from keras.applications import Xception

#Caminhos
dataset = "Dataset"
treino = "treino"
teste = "teste"
validacao = "validacao"

#Variáveis
nparticles = 1
gsa_iter = 1
progresso = 0
nrimagens = 13315 #atualmente incorreto
larg = 299
alt = 299
num_classes = 3
atv = 'relu'

columns = ['nepochs', 'nneurons1', 'nneurons2', 'accuracy']
excel = pd.DataFrame(columns=columns)

x_test = []
y_test = []

#Cria lista com as classes presentes na pasta
classes = os.listdir(dataset)

#Cria Dataframe
df = pd.DataFrame(columns=['dir_imagem', 'classe'])

#Vai a todas as imagens e coloca-as no dataframe com a classe correspondente
for c in classes:
    dir_classe = os.path.join(dataset, c)
    for nome_img in os.listdir(dir_classe):
        df.loc[len(df)] = [os.path.join(dir_classe, nome_img), c]


#divide o dataframe entre treino, validação e teste (80% treino, 10% validação, 10% teste)
df_treino, df_temp = train_test_split(df, test_size=0.2, random_state=1)
df_validacao, df_teste = train_test_split(df_temp, test_size=0.5, random_state=1)

# Apaga as pastas treino, validacao e teste e os conteúdos
shutil.rmtree(treino, ignore_errors=True)
shutil.rmtree(validacao, ignore_errors=True)
shutil.rmtree(teste, ignore_errors=True)

# Cria as pastas treino, validacao e teste, e em cada uma delas, uma pasta para cada classe, e preenche-as
for c in classes:
    fsn_treino = os.path.join(treino, c)
    fsn_validacao = os.path.join(validacao, c)
    fsn_teste = os.path.join(teste, c)
    os.makedirs(fsn_treino, exist_ok=True)
    os.makedirs(fsn_validacao, exist_ok=True)
    os.makedirs(fsn_teste, exist_ok=True)
    percentagem = (progresso / nrimagens) * 100
    #print(f"Progresso: {percentagem:.2f}%")
    for dir_atual, classe_imagem in df_treino[df_treino["classe"] == c].values:
        imagem = dir_atual.split("\\")[-1]
        dir_novo = os.path.join(fsn_treino, imagem)
        shutil.copy(dir_atual, dir_novo)
        # Retirar para mais performance
        #print(f"Progresso: {percentagem:.2f}%")
        progresso += 1
        percentagem = (progresso / nrimagens) * 100

    for dir_atual, classe_imagem in df_validacao[df_validacao["classe"] == c].values:
        imagem = dir_atual.split("\\")[-1]
        dir_novo = os.path.join(fsn_validacao, imagem)
        shutil.copy(dir_atual, dir_novo)
        # Retirar para mais performance
        #print(f"Progresso: {percentagem:.2f}%")
        progresso += 1
        percentagem = (progresso / nrimagens) * 100

    for dir_atual, classe_imagem in df_teste[df_teste["classe"] == c].values:
        imagem = dir_atual.split("\\")[-1]
        dir_novo = os.path.join(fsn_teste, imagem)
        shutil.copy(dir_atual, dir_novo)
        # Retirar para mais performance
        #print(f"Progresso: {percentagem:.2f}%")
        progresso += 1
        percentagem = (progresso / nrimagens) * 100
        
#Vai a uma pasta, e automaticamente retira as imagens e a sua classe
ds_treino = image_dataset_from_directory(treino, batch_size=32, image_size=(larg,alt))
ds_teste = image_dataset_from_directory(teste, batch_size=32, image_size=(larg,alt))
ds_validacao = image_dataset_from_directory(validacao, batch_size=32, image_size=(larg, alt))

#Divide ds_teste
for images, labels in ds_teste:
    x_test.append(images)
    y_test.append(labels)


x_test = tf.concat(x_test, axis=0)
y_test = tf.concat(y_test, axis=0)

usar_modelo_anterior = input("Deseja usar um modelo treinado anteriormente? (s/n): ").lower()

def train_and_evaluate(params, x = 0):
    global flag
    global excel
    nepochs = int(params[0])
    nneurons1 = int(params[1])
    nneurons2 = int(params[2])
    try:
            #tipo de modelo

        base_model = Xception(weights='imagenet')

            # Cria um modelo sequencial
        modelo = tf.keras.Sequential()

            # Adiciona a base da Xception ao modelo
        modelo.add(base_model)
    
        modelo.add(tf.keras.layers.Flatten(input_shape=(larg, alt,3)))  #input
    
        modelo.add(tf.keras.layers.Dense(nneurons1, activation=atv))  #hidden
        modelo.add(tf.keras.layers.Dense(nneurons2, activation=atv))  #hidden

        modelo.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  #output
    
    
            #TREINO
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            #treina o modelo nepochs vezes
        modelo.fit(ds_treino, epochs=nepochs, batch_size=32, validation_data=ds_validacao)

            #print(modelo.evaluate(ds_teste))
        if int(x) == 1:
            nome = input("Qual o nome do modelo que deseja guardar(.h5 necessario!!)? ").lower()
            modelo.save(nome)
         
        #O modelo faz as previsões para as imagens teste
        y_pred = modelo.predict(x_test)

        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
    
        excel = excel.append({'nepochs': nepochs, 'nneurons1': nneurons1, 'nneurons2': nneurons2, 'accuracy': accuracy}, ignore_index=True)

        #Resultados da previsão
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        result = classification_report(y_test, y_pred_classes, zero_division=1)

    
        aucovr = (roc_auc_score(y_test, y_pred, multi_class='ovr'))
        aucovo = (roc_auc_score(y_test, y_pred, multi_class='ovo'))
    
        sens, spec, _ , _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted', zero_division=1)
        fmeasure = 2*(sens * spec)/(sens + spec)
    
        global best_nepochs, best_nneurons1, best_nneurons2, best_spec, best_sens, best_accuracy, best_aucovr, best_aucovo, best_conf_matrix, best_result, best_fmeasure
        best_nepochs, best_nneurons1, best_nneurons2, best_spec, best_sens, best_accuracy, best_aucovr, best_aucovo, best_conf_matrix, best_result, best_fmeasure = nepochs, nneurons1, nneurons2, spec, sens, accuracy, aucovr, aucovo, conf_matrix, result, fmeasure
    
        return -accuracy
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        excel.to_excel('output.xlsx', index=False)

if usar_modelo_anterior != 's':
    lb = np.array([1, 16, 16])
    ub = np.array([1, 128,128]) 


    #Executar a otimização GSA
    alggsa  = spp.gsa(nparticles,train_and_evaluate,lb,ub,3,gsa_iter)

    best_gsa = alggsa.get_Gbest()

    alh  = spp.pso(nparticles,train_and_evaluate,lb,ub,3,gsa_iter)

    best_pso = alh.get_Gbest()

    print(f"\nResultado para os melhores hiperparâmetros encontrados pelo GSA: {best_gsa}")

    print(f"\nResultado para os melhores hiperparâmetros encontrados pelo PSO: {best_pso}")

    excel.to_excel('output.xlsx', index=False)
    train_and_evaluate(best_gsa,1)
    print("REDE(GSA)")
    print("\nCamada 2 :\n número de neuronios -> " + str(best_nneurons1) + "\n função de ativação -> " + atv)
    ("\nCamada 3 :\n número de neuronios -> " + str(best_nneurons2) + "\n função de ativação -> " + atv)

    print("\nCamada 4:\n número de neuronios ->" + str(num_classes) + "\n função de ativação -> softmax")
   
    print(f"\nEspecificidade: {best_spec:.4f}")
    print(f"Sensibilidade: {best_sens:.4f}")
    print(f"f-measure: {best_fmeasure:.4f}")
    print(f"accuracy: {best_accuracy:.4f}")

    print(f"\nAUC (OvR): {best_aucovr:.4f}")
    print(f"AUC (OvO): {best_aucovo:.4f}")

    print("\nMatriz de Confusão:")
    print(best_conf_matrix)
    
    print("\nRelatório de Classificação:")
    print(best_result)

    train_and_evaluate(best_pso,1)
    print("REDE(PSO)")
    print("\nCamada 2 :\n número de neuronios -> " + str(best_nneurons1) + "\n função de ativação -> " + atv)
    print("\nCamada 3 :\n número de neuronios -> " + str(best_nneurons2) + "\n função de ativação -> " + atv)

    print("\nCamada 4:\n número de neuronios ->" + str(num_classes) + "\n função de ativação -> softmax")
   
    print(f"\nEspecificidade: {best_spec:.4f}")
    print(f"Sensibilidade: {best_sens:.4f}")
    print(f"f-measure: {best_fmeasure:.4f}")
    print(f"accuracy: {best_accuracy:.4f}")

    print(f"\nAUC (OvR): {best_aucovr:.4f}")
    print(f"AUC (OvO): {best_aucovo:.4f}")

    print("\nMatriz de Confusão:")
    print(best_conf_matrix)
    
    print("\nRelatório de Classificação:")
    print(best_result)
else:
    try:
        nome = input("Qual o nome do modelo (.h5)? ").lower()
        modelo = tf.keras.models.load_model(nome)

        y_pred = modelo.predict(x_test)

        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)

        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        result = classification_report(y_test, y_pred_classes, zero_division=1)

        aucovr = (roc_auc_score(y_test, y_pred, multi_class='ovr'))
        aucovo = (roc_auc_score(y_test, y_pred, multi_class='ovo'))

        sens, spec, _, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted', zero_division=1)
        fmeasure = 2 * (sens * spec) / (sens + spec)

        print("Resultados do modelo carregado:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Especificidade: {spec:.4f}")
        print(f"Sensibilidade: {sens:.4f}")
        print(f"f-measure: {fmeasure:.4f}")

        print(f"AUC (OvR): {aucovr:.4f}")
        print(f"AUC (OvO): {aucovo:.4f}")

        print("Matriz de Confusão:")
        print(conf_matrix)

        print("Relatório de Classificação:")
        print(result)

    except Exception as e:
        print(f"Erro: {e}")
    

