import os
import subprocess
import sys
import webbrowser
import time

def main():
    # Path to the tensorboard logs
    # Based on findings: agents/dqn/entrenamiento_10/RESULTADOS_BREAKOUT_10M/entrenamiento_breakout_10M/tensorboard
    log_dir = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'agents', 'dqn', 'entrenamiento_10', 'RESULTADOS_BREAKOUT_10M', 'entrenamiento_breakout_10M', 'tensorboard'
    )
    
    log_dir = os.path.abspath(log_dir)
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found at {log_dir}")
        return

    print(f"Starting TensorBoard with logdir={log_dir}")
    print("Opening browser...")
    
    # Construct command
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir]
    
    try:
        # Open browser after a short delay to allow TB to start
        # Default port is 6006
        url = "http://localhost:6006/"
        
        # Start TensorBoard
        # We use Popen to run it in the background/parallel process
        process = subprocess.Popen(cmd)
        
        print(f"TensorBoard started. Navigate to {url}")
        print("Press Ctrl+C to stop.")
        
        time.sleep(2)
        webbrowser.open(url)
        
        # Wait for process to finish (user stops it)
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
