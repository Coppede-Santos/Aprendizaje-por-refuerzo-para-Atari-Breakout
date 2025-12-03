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
