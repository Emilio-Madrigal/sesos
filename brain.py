import tensorflow as tf
import numpy as np
import os
from core.ann import ANN
import data.dataset_and as AND

def guardar_modelo(modelo, nombre_archivo="modelo_entredado"):
    try:
        os.makedirs("modelos_guardados", exist_ok=True)
        modelo_datos={
            'estructura':[],
            'activaciones':[],
            'pesos':[],
            'bias':[],
            'loss_list':modelo.loss_list
        }

        for i, layer in enumerate(modelo.layers):
            modelo_datos['estructura'].append(layer.cantidad_neuronas)
            if hasattr(layer.neuronas[0],'activacion_nombre'):
                modelo_datos['activaciones'].append(layer.neuronas[0].activacion_nombre)
            else:
                modelo_datos['activaciones'].append('sigmoide')

            pesos_capa=[]
            bias_capa=[]
            for neurona in layer.neuronas:
                pesos_capa.append(neurona.pesos.numpy())
                bias_capa.append(neurona.bias.numpy())
            
            modelo_datos['pesos'].append(pesos_capa)
            modelo_datos['bias'].append(bias_capa)

        if modelo.layers:
            modelo_datos['estructura'].insert(0,modelo.layers[0].cantidad_entradas)
        
        ruta=f"modelos_guardados/{nombre_archivo}.npz"
        np.savez(ruta, **modelo_datos)
        print(f"ya esta aqui esta: {ruta}")
    except Exception as e:
        print(f"no se guardo :( {e}")

def cargar_modelo(nombre_archivo):
    try:
        ruta=f"modelos_guardados/{nombre_archivo}.npz"

        if not os.path.exists(ruta):
            print(f"no habia nada en esta ruta: {ruta}")
            return None
        
        datos=np.load(ruta,allow_pickle=True)
        estructura=datos['estructura'].tolist()
        activaciones=datos['activaciones'].tolist()

        modelo=ANN(estructura,activaciones)

        pesos_g=datos['pesos']
        bias_g=['bias']

        for i, layer in enumerate(modelo.layers):
            for j, neurona in enumerate(layer.neuronas):
                neurona.pesos.assign(pesos_g[i][j])
                neurona.bias.assign(bias_g[i][j])
        
        if 'loss_list' in datos:
            modelo.loss_list=datos['loss_list'].toList()
        
        print(f"modelo cargado")
        return modelo
    except Exception as e:
        print(f"no se cargo el modelo :( {e}")
        return None

def entrenar_modelo():
    print("entrenando...")
    
    num_entradas = len(AND.entradas[0])
    num_salidas = len(AND.salidas[0])

    estructura = [num_entradas, 3, num_salidas]
    activaciones = ['relu', 'sigmoide']  # Ahora sí funciona
    
    modelo = ANN(estructura, activaciones)
    modelo.compile_with_optimizer('adam', 0.001)
    
    print(f"estructura: {estructura}")
    print(f"activaciones: {activaciones}")
    print(f"datos de entrenamiento: {len(AND.entradas)} muestras")

    epochs = int(input("\nnumero de epocas: ") or "1500")
    
    X_tensor = tf.constant(AND.entradas, dtype=tf.float32)
    y_tensor = tf.constant(AND.salidas, dtype=tf.float32)
    
    for epoch in range(epochs):
        loss = modelo.train_step_tf(X_tensor, y_tensor)
        modelo.loss_list.append(float(loss))
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoca {epoch+1:4d}: perdida = {loss:.6f}")
    
    print("se acabo")
    return modelo

def probar_modelo(modelo):
    print("\nresultados del dataset de entrenamiento:")
    print("Entrada A | Entrada B | Esperado | Predicho | Error")
    print("-" * 50)
    
    error_total = 0
    for entrada, esperado in zip(AND.entradas, AND.salidas):
        entrada_tf = tf.constant(entrada, dtype=tf.float32)
        prediccion = modelo.forward(entrada_tf)
        
        pred_valor = float(prediccion.numpy())
        error = abs(esperado[0] - pred_valor)
        error_total += error
        
        print(f"    {entrada[0]}     |     {entrada[1]}     |    {esperado[0]}     |  {pred_valor:.4f}  | {error:.4f}")
    
    error_promedio = error_total / len(AND.entradas)
    print(f"\nError promedio: {error_promedio:.4f}")
    
    umbral = 0.1
    if error_promedio < umbral:
        print("Modelo entrenado correctamente")
    else:
        print("El modelo necesita más entrenamiento")
    
    # Prueba manual simple
    print("\nprueba la compuerta AND:")
    while True:
        try:
            entrada_str = input("entrada (A B, ej: '1 0') o 'salir': ").strip().lower()
            
            if entrada_str in ['salir', 'exit', 'quit', 's']:
                break
                
            valores = entrada_str.split()
            if len(valores) != 2:
                print("ingresa dos valores ")
                continue
                
            a, b = int(valores[0]), int(valores[1])
            
            if a not in [0, 1] or b not in [0, 1]:
                print("los valores deben ser 0 o 1")
                continue
            
            entrada_tf = tf.constant([a, b], dtype=tf.float32)
            prediccion = modelo.forward(entrada_tf)
            resultado = float(prediccion.numpy())
            resultado_binario = 1 if resultado > 0.5 else 0
            
            esperado_and = 1 if (a == 1 and b == 1) else 0
            correcto = "correcto" if resultado_binario == esperado_and else "estas mal"
            
            print(f"A={a}, B={b} -> Salida: {resultado_binario} ({resultado:.4f}) - {correcto}")
            
        except (ValueError, KeyboardInterrupt):
            break

def main():
    respuesta = input("cargar modelo existente? (s/n): ").lower().strip()
    
    modelo = None
    if respuesta == 's':
        nombre = input("nombre del modelo: ").strip()
        modelo = cargar_modelo(nombre)

    if modelo is None:
        print("\nentrenando nuevo modelo...")
        modelo = entrenar_modelo()

        guardar = input("\nguardar el nuevo modelo? (s/n): ").lower().strip()
        if guardar == 's':
            nombre = input("nombre del modelo: ").strip()
            if not nombre:
                nombre = "modelo_entrenado"
            guardar_modelo(modelo, nombre)
    
    probar_modelo(modelo)


    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(modelo.loss_list)
    plt.title('perdida durante el entrenamiento')
    plt.xlabel('epoca')
    plt.ylabel('perdida')
    plt.grid(True)
    plt.show()
    
    print(f"\nperdida final: {modelo.loss_list[-1]:.6f}")

if __name__ == "__main__":
    main()