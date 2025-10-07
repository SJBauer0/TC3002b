# Red de Hopfield con Números Mayas

## Diagnóstico: Falla en el Reconocimiento de Patrones

A pesar de una implementación correcta del algoritmo, la red a menudo no logra reconstruir los patrones memorizados, especialmente al almacenar más de dos o tres de ellos.

La causa **no es un error de código**, sino una limitación del modelo de Hopfield con los datos de entrada seleccionados (Numeros Maya).

### Causa Principal: Alta Correlación entre Patrones

El modelo de Hopfield funciona con patrones muy distintos entre sí. Los números mayas, sin embargo, son altamente **parecidos** entre si.

Este alto grado de similitud cause memorias incorrectas que son una mezcla de los patrones originales y la red queda atrapada durante el proceso de reconocimiento.

Como resultado, la **capacidad de almacenamiento efectiva** de la red se reduce y queda debajo del límite teórico ideal.

---

## Conclusión

Este proyecto sirve como una demostración práctica de las limitaciones teóricas del modelo de Hopfield. El código es funcional y ejecuta el algoritmo correctamente, pero los datos de entrada impide que la red opere de manera correcta.

---

## Ejecutar

1.  Asegúrate de que los archivos `uno_maya.txt`, `dos_maya.txt`, etc., estén en el mismo directorio.
2.  Ejecuta el script desde la terminal:
    ```bash
    python3 app.py
    ```
