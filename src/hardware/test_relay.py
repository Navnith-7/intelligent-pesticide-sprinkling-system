import serial
import time

PORT = "COM6"
BAUD_RATE = 9600

print("Connecting to Arduino...")

arduino = serial.Serial(PORT, BAUD_RATE, timeout=2)

# Arduino resets when serial connection opens
time.sleep(2)

arduino.reset_input_buffer()

print("Connected successfully!")

try:
    while True:
        command = input("\nEnter 1 for ON, 0 for OFF, q to quit: ").strip()

        if command == "1":
            arduino.write(b"1")
            print("Sent ON command")

        elif command == "0":
            arduino.write(b"0")
            print("Sent OFF command")

        elif command.lower() == "q":
            arduino.write(b"0")
            print("Sent OFF command. Exiting.")
            break

        else:
            print("Invalid input. Enter 1, 0, or q.")

        time.sleep(0.2)

        response = arduino.readline().decode(
            "utf-8", errors="ignore"
        ).strip()

        if response:
            print("Arduino:", response)

finally:
    arduino.write(b"0")
    arduino.close()
    print("Serial connection closed.")