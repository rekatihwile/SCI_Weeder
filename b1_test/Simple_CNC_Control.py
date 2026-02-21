import Laser_Helpers as lh
import time

def main():
    ser = lh.connect() 
    print("\n--- SCI_Weeder Laser Debug Utility ---")
    print("Commands: 'x,y', 'fire P T', 'home', 'raw [command]', 'q'")

    try:
        while True:
            user_input = input("\nEnter Command: ").strip()

            if user_input.lower() == 'q':
                break

            elif user_input.lower().startswith('raw'):
                raw_cmd = user_input[3:].strip()
                if raw_cmd:
                    print(f"Sending: {raw_cmd}")
                    print(f"Response: {lh.send(ser, raw_cmd)}")

            elif user_input.lower() == 'home':
                print("Homing...")
                print(f"Homing Response: {lh.send(ser, '$H')}")
                lh.wait_for_idle(ser)

            elif ',' in user_input:
                try:
                    x_str, y_str = user_input.split(',')
                    target_x, target_y = float(x_str), float(y_str)
                    print(f"Moving to {target_x}, {target_y}...")
                    
                    response = lh.move_to(ser, target_x, target_y)
                    
                    if response and ("ALARM" in response or "error" in response.lower()):
                        print(f"🚨 MACHINE FAULT: {response}")
                    else:
                        print(f"Success: {response}")
                except ValueError:
                    print("Invalid format. Use: x,y")

    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        lh.close(ser)

if __name__ == "__main__":
    main()