import Laser_Helpers as lh

def main():
    # 1. Establish connection and initialize GRBL settings
    ser = lh.connect() 
    print("\n--- Laser Test Utility (New Head Edition) ---")
    print("Commands:")
    print("  'x,y'      : Move to coordinate (e.g., 100,150)")
    print("  'fire P T' : Pulse laser. P=Power (0-100), T=Time in seconds")
    print("               Example: 'fire 50 0.5' for 50% power for 0.5s")
    print("  'home'     : Re-home the machine")
    print("  'q'        : Quit and close connection")

    try:
        while True:
            user_input = input("\nEnter Command: ").strip().lower()

            if user_input == 'q':
                break

            elif user_input == 'home':
                print("Homing...")
                lh.send(ser, "$H")
                lh.wait_for_idle(ser)
                print("Homing complete.")

            elif user_input.startswith('fire'):
                try:
                    # Split 'fire P T' into components
                    parts = user_input.split()
                    
                    if len(parts) == 3:
                        # Convert percentage (0-100) to GRBL scale (0-1000)
                        power_percent = float(parts[1])
                        grbl_power = int((power_percent / 100.0) * 1000)
                        
                        duration = float(parts[2])
                        
                        # Validate inputs to protect the new head
                        grbl_power = max(0, min(1000, grbl_power))
                        duration = max(0.001, duration)

                        print(f"🔥 Firing: {power_percent}% Power ({grbl_power}) for {duration}s...")
                        lh.burn(ser, power=grbl_power, duration=duration)
                    else:
                        print("Invalid format. Use: fire [Power%] [Time]")
                        print("Example: fire 80 1.5")
                except ValueError:
                    print("Error: Power and Time must be numbers.")

            elif ',' in user_input:
                try:
                    x_str, y_str = user_input.split(',')
                    target_x, target_y = float(x_str), float(y_str)
                    
                    print(f"Moving to X:{target_x}, Y:{target_y}...")
                    lh.move_to(ser, target_x, target_y)
                except ValueError:
                    print("Invalid coordinate format. Use x,y (e.g., 50,50)")

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        lh.close(ser)

if __name__ == "__main__":
    main()