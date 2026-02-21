import serial
import threading
import sys
import time

# --- CONFIGURATION ---
# Change this to your laser's port. 
# Windows: 'COM3', 'COM4', etc.
# Mac/Linux: '/dev/ttyUSB0', '/dev/tty.usbserial', etc.
SERIAL_PORT = '/dev/ttyUSB1'  
BAUD_RATE = 115200

def read_from_port(ser):
    """Continuously listen to the serial port and print incoming data."""
    while True:
        try:
            if ser.in_waiting > 0:
                # Read the line, decode it, and strip whitespace
                response = ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    print(f"\n[LASER] {response}")
                    print(">> ", end="", flush=True) # Reprint prompt
        except Exception as e:
            print(f"\n[!] Error reading from serial: {e}")
            break

def main():
    try:
        # Open the serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[*] Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        
        # Wake up Grbl by sending a few newline characters
        ser.write(b"\r\n\r\n")
        time.sleep(2)
        ser.reset_input_buffer()
        
    except serial.SerialException as e:
        print(f"[!] Could not open port {SERIAL_PORT}: {e}")
        print("Ensure the laser is plugged in, powered on, and no other software is using the port.")
        sys.exit(1)

    # Start the background thread to handle incoming responses
    listener_thread = threading.Thread(target=read_from_port, args=(ser,), daemon=True)
    listener_thread.start()

    print("[*] Two-way serial terminal active.")
    print("[*] Type your Grbl commands (e.g., $$ or $H) and press Enter.")
    print("[*] Type 'exit' or 'quit' to close the connection.\n")

    # Main loop for sending commands
    while True:
        try:
            command = input(">> ")
            
            if command.lower() in ['exit', 'quit']:
                print("[*] Closing connection...")
                break
            
            # Grbl strictly requires commands to end with a newline or carriage return
            formatted_command = command + '\n'
            ser.write(formatted_command.encode('utf-8'))
            
        except KeyboardInterrupt:
            print("\n[*] Exiting via keyboard interrupt...")
            break

    ser.close()
    print("[*] Terminal closed.")

if __name__ == '__main__':
    main()