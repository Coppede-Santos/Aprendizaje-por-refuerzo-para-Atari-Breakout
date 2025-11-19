# Entorno Atari — Breakout

Este documento resume cómo funciona el entorno Atari Breakout en Gymnasium (usando ALE), con foco en sus observaciones, acciones, recompensas y modos de renderizado, así como su configuración.

## 1. Visión general

Los entornos Atari en Gymnasium están implementados a través del Arcade Learning Environment (ALE), que emula juegos de Atari 2600 mediante Stella.

Para usar los entornos, hay que tener instalados los ROMs apropiados. Gymnasium sugiere instalar con:
 ```
pip install gymnasium[atari]
 ```

Esto instala ale-py y los ROMs necesarios. 


En versiones recientes, para poder crear entornos Atari hay que importar ale_py:

 ```
import gymnasium as gym  
import ale_py  
env = gym.make("ALE/Breakout-v5")
 ```

Para la visualización, se puede usar el argumento render_mode al crear el entorno: modos disponibles son human y rgb_array.

## 2. Configuración del entorno (v5 recomendado)

Se recomienda usar la versión v5 del entorno (ALE/Breakout-v5), que sigue las mejores prácticas actuales. 

### Parámetros configurables al crear el entorno:

* obs_type: tipo de observación. Puede ser "rgb", "grayscale" o "ram" 

* frameskip: controla cuántos frames se saltean (frame-skipping). Puede ser un entero o una tupla para definir un rango.

* repeat_action_probability: probabilidad de que la acción anterior se “pegue” (sticky actions) en vez de ejecutar la nueva, para introducir estocasticidad. 

* full_action_space: booleano. Si es True, se usa el espacio completo de acciones legales de Atari; si es False, se usa un subconjunto.

* difficulty y mode: para definir un “flavor” del entorno según la dificultad y modo de juego.

### Valores por defecto / típicos para Breakout-v5:

* frameskip: 5 (stochastic frame-skipping desactivado en v5)

* repeat_action_probability: 0.25 (sticky actions)

* full_action_space: True por defecto en v5. 


## 3. Espacio de acciones

En la versión por defecto de Breakout (sin full_action_space=True), el espacio de acciones es discreto con 4 acciones:

| Valor | Acción                           |
|-------|-----------------------------------|
|   0   | NOOP (no hacer nada)              |
|   1   | FIRE (disparar para lanzar la bola) |
|   2   | RIGHT (mover la paleta a la derecha) |
|   3   | LEFT (mover la paleta a la izquierda) |


	

Si se activa full_action_space=True, se pueden usar todas las acciones legales de Atari 2600, que son hasta 18 acciones (direcciones + fire, combinaciones, etc.). 

También es posible usar un espacio de acción continuo usando ALE con continuous=True: en ese caso, la acción se representa como un vector (por ejemplo, polar + fire) con valores reales. 

## 4. Observaciones

Breakout (y otros entornos Atari) pueden devolver distintos tipos de observaciones:

* obs_type="rgb": retorna la imagen en color que vería un jugador humano. El espacio de observación es Box(0, 255, (210, 160, 3), uint8). 

* obs_type="grayscale": versión en escala de grises de la imagen. El espacio es Box(0, 255, (210, 160), uint8). 

* obs_type="ram": retorna el estado de la memoria RAM de la consola Atari (128 bytes). El espacio es Box(0, 255, (128,), uint8). 

## 5. Recompensas

En Breakout, las recompensas se obtienen al destruir ladrillos del muro. 
gymnasium.farama.org

El valor de la recompensa depende del color del ladrillo destruido. 

No hay una recompensa negativa explícita por perder vidas (al menos no documentada como penalización aparte), el puntaje se ajusta según los puntos que el juego define.

## 6. Estocasticidad y dinámica del entorno

* Sticky actions: ALE implementa “sticky actions”, es decir, con cierta probabilidad (repeat_action_probability) la acción anterior se repite en lugar de la nueva. Esto introduce estocasticidad y evita que los agentes simplemente memoricen secuencias fijas de acciones. 


* Frame skipping: Gymnasium agrega “frame skipping” estocástico por defecto. Es decir, cuando ejecutás env.step(action), esa acción se repite por varios fotogramas (frames), para simular cómo los juegos Atari funcionan realmente y para reducir la carga de pasos. 


## 7. Renderizado / visualización

Al crear el entorno, se puede indicar render_mode para definir cómo se va a dibujar / mostrar la pantalla:

* human: renderiza la pantalla para que la pueda ver un humano, con buena escala y audio. 

* rgb_array: no abre una ventana, pero devuelve un array RGB (frame) en el diccionario de info en cada step, lo que permite capturar los frames para grabarlos o procesarlos. 

Se recomienda especificar render_mode al crear el entorno (y no usar solo env.render()), para garantizar que la visualización funcione correctamente, con audio y escalado adecuados. 


## 8. ALE (Arcade Learning Environment)

ALE es la capa subyacente que emula el Atari 2600 usando Stella.

Se puede usar directamente la interfaz de ALE a través de ale_py.ALEInterface(): esto permite cargar un ROM, resetear el juego, tomar acciones, obtener la pantalla, etc. 
PyPI

Gymnasium registra los entornos ALE para usarlos como entornos estándar de RL.

## 9. “Flavors” del entorno Breakout

Breakout tiene distintos “sabores” (“flavors”) definidos por combinaciones de mode y difficulty.

En la documentación oficial para Breakout: los modos válidos son [0, 4, 8, 12, …, 44] (lista predeterminada) y las dificultades válidas son [0, 1]. 


Elegir un sabor distinto puede cambiar la dinámica del juego (velocidad, comportamiento, etc.).

## 10. Resumen de recomendaciones para entrenar agentes

Usar la versión ALE/Breakout-v5, ya que está alineada con las mejores prácticas actuales.

Definir explícitamente los parámetros obs_type, frameskip, repeat_action_probability y full_action_space para tener control sobre el comportamiento del entorno.

Para agentes basados en visión (CNNs), usar obs_type="rgb" o "grayscale".

Si querés usar una versión más “compacta” del estado, usar obs_type="ram", aunque eso implica perder la representación visual.

Para grabar o procesar los frames, usar render_mode="rgb_array".

Tener en cuenta la estocasticidad por sticky actions y frame skipping al diseñar la política de entrenamiento: puede afectar el rendimiento, exploración y estabilidad del agente.

## Fuentes

* https://gymnasium.farama.org/v1.0.0a2/environments/atari/
* https://ale.farama.org/main/gymnasium-interface/
* https://gymnasium.farama.org/v1.0.0a2/environments/atari/breakout/