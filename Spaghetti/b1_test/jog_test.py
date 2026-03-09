import time
import numpy as np
from SCI_Weeder.b1_test.PID.helpers import LaserHelper

# ===========================
# Configuration
# ===========================
PORT = "/dev/ttyUSB0"
BAUD = 115200

# Workspace Limits (The physical bounds you want to cover)
# Adjust these to your Longer B1's actual max travel
X_MIN, X_MAX = 20.0, 430.0  
Y_MIN, Y_MAX = 20.0, 420.0

# Desired Overlap (30% of FOV)
FOV_W, FOV_H = 120.0, 100.0
TARGET_X_STEP = FOV_W * 0.3  # ~36mm
TARGET_Y_STEP = FOV_H * 0.3  # ~30mm

def run_edge_to_edge_scan():
    bot = LaserHelper(PORT, BAUD)
    
    try:
        print("--- PHASE 1: INITIALIZING ---")
        bot.send_command("$H") # Home first
        bot.wait_for_idle()

        # Calculate exact steps to cover the limits
        # We use 'ceil' to ensure we have ENOUGH steps, then linspace to even them out
        num_x_steps = int(np.ceil((X_MAX - X_MIN) / TARGET_X_STEP)) + 1
        num_y_steps = int(np.ceil((Y_MAX - Y_MIN) / TARGET_Y_STEP)) + 1
        
        # Generate the exact coordinate arrays (Includes the Max Limits!)
        x_coords = np.linspace(X_MIN, X_MAX, num_x_steps)
        y_coords = np.linspace(Y_MAX, Y_MIN, num_y_steps) # Scanning top to bottom

        print(f"Grid Configured: {num_x_steps} columns x {num_y_steps} rows")
        print(f"Total Viewpoints: {num_x_steps * num_y_steps}")
        
        direction = 1 # Snake toggle
        
        for row_idx, y in enumerate(y_coords):
            # Reverse X array for the snake pattern
            current_row_x = x_coords if direction == 1 else x_coords[::-1]
            
            print(f"\nScanning Row {row_idx + 1}/{num_y_steps} (Y={y:.2f})")
            
            for x in current_row_x:
                print(f"  -> Viewpoint: X{x:.2f}, Y{y:.2f}")
                
                bot.move_to(x, y, speed=10000)
                bot.wait_for_idle()
                
                # --- CV PLACEHOLDER ---
                # This is where your black dot matching runs
                time.sleep(0.3) 
            
            direction *= -1 # Flip for next row

        print("\n--- FULL COVERAGE COMPLETE ---")
        bot.move_to(0, 0, speed=12000)

    except KeyboardInterrupt:
        bot.stop_motion()
        bot.send_command("M5")
    finally:
        bot.close()

if __name__ == "__main__":
    run_edge_to_edge_scan()