import requests

def obtener_indicadores_esios():
    # URL de la ruta indicada en la documentación
    url = "https://api.esios.ree.es/indicators"
    
    # Cabeceras (Headers) requeridas por la API
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": "YOUR_ESIOS_API_KEY_HERE"
    }
    
    # Parámetros de la URL. La documentación menciona "locale" (es, en)
    params = {
        "locale": "es" # Puedes cambiarlo a "en" si prefieres los nombres en inglés
    }

    try:
        # Realizamos la petición GET
        respuesta = requests.get(url, headers=headers, params=params)
        
        # Lanzamos una excepción si el servidor devuelve un código de error (ej. 401, 404, 500)
        respuesta.raise_for_status()
        
        # Parseamos el cuerpo (Body) de la respuesta a un diccionario de Python
        datos = respuesta.json()
        
        # Extraemos la lista de indicadores
        indicadores = datos.get("indicators", [])
        
        # Mostramos un pequeño resumen
        print(f"Conexión exitosa. Se han encontrado {datos.get('meta', {}).get('size', len(indicadores))} indicadores en total.\n")
        
        # Imprimimos los 10 primeros indicadores como muestra
        print("Mostrando los primeros 10 indicadores:")
        print("-" * 40)
        for indicador in indicadores:
            print(f"{indicador.get('id')} {indicador.get('name')}")
            
        return indicadores

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API de ESIOS: {e}")
        return None

if __name__ == "__main__":
    lista_de_indicadores = obtener_indicadores_esios()
