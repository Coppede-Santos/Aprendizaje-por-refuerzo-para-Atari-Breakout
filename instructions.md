1. Crear entorno virtual de python e instalar dependencias

   ```bash
    python.exe -m pip install --upgrade pip
    pip install -r requirements.txt
   ```
2. Activar el entorno virtual

   ```bash
    .venv\Scripts\Activate.ps1
   ```

3. Correr el script de prueba
   ```bash
    python env/test_env.py
   ```

4. Recolección de Métricas

   Puedes recolectar métricas para todos los agentes o agentes específicos usando el script `collect_metrics.py`.

   **Uso básico (correr todos los agentes):**
   ```bash
   python metrics/collect_metrics.py
   ```

   **Agentes Disponibles:**
   - `Random`
   - `FollowBall`
   - `QLearning`
   - `DQN`

   **Opciones:**
   - `--agents`: Lista de agentes a ejecutar (por defecto corre todos).
   - `--episodes`: Número de episodios a correr por agente (por defecto: 10).
   - `--render`: Habilitar renderizado para ver al agente jugar.

   **Ejemplos:**

   Correr todos los agentes por 10 episodios:
   ```bash
   python metrics/collect_metrics.py
   ```

   Correr solo Random y DQN por 5 episodios y ver cómo juegan:
   ```bash
   python metrics/collect_metrics.py --agents Random DQN --episodes 5 --render
   ```

5. Visualizar Métricas en TensorBoard

   Para ver las métricas de entrenamiento del agente DQN (loss, reward promedio, etc.):

   ```bash
   python metrics/view_tensorboard.py
   ```

   Esto abrirá automáticamente una ventana del navegador con TensorBoard.

6. Generación de Gráficos de Métricas

   Puedes generar gráficos comparativos (boxplots) de las métricas recolectadas:

   ```bash
   python metrics/graphs.py
   ```

   Esto generará los siguientes gráficos en la carpeta `metrics/plots/`:
   - `total_reward_boxplot.png`
   - `steps_boxplot.png`
   - `brick_hits_boxplot.png`

   **Colores de los Agentes:**
   - Random: Gris
   - FollowBall: Amarillo
   - QLearning: Verde
   - DQN: Rojo

7. Exportar Gráficos de TensorBoard

   Si deseas guardar los gráficos de TensorBoard como imágenes (PNG) sin usar el navegador:

   ```bash
   python metrics/export_tensorboard.py
   ```

   Los gráficos se guardarán en `metrics/plots/tensorboard/`.

8. Grabar Videos de las Partidas

   Puedes grabar un video (GIF) de una partida completa para cada agente:

   ```bash
   python metrics/record_videos.py
   ```

   Los videos se guardarán en `metrics/videos/` como archivos `.gif`.

