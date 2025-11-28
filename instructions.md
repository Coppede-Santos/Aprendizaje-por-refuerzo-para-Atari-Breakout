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

   Puedes recolectar métricas para diferentes agentes usando el script `collect_metrics.py`.

   **Uso básico:**
   ```bash
   python metrics/collect_metrics.py --agent <NombreDelAgente>
   ```

   **Agentes Disponibles:**
   - `RandomAgent` (por defecto)
   - `FollowBallAgent`

   **Opciones:**
   - `--agent`: Nombre del agente a ejecutar.
   - `--episodes`: Número de episodios a correr (por defecto: 10).
   - `--render`: Habilitar renderizado para ver al agente jugar.
   - `--output`: Ruta al archivo CSV de salida (por defecto: `metrics/<NombreDelAgente>.csv`).

   **Ejemplos:**

   Correr RandomAgent por 10 episodios:
   ```bash
   python metrics/collect_metrics.py --agent R
   ```

   Correr FollowBallAgent por 5 episodios y ver cómo juega:
   ```bash
   python metrics/collect_metrics.py --agent F --episodes 5 --render
   ```
