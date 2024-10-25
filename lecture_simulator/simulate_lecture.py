import time
import random
from datetime import timedelta


def simulate_lecture_updates(input_file, output_file):
    # Starting time at 0:00
    current_time = timedelta(seconds=0)

    # Open the input file and output file
    with open(input_file, 'r') as lecture_file, open(output_file, 'w') as output:
        for line in lecture_file:
            # Strip the line of any leading/trailing whitespace
            line = line.strip()

            if line:  # Ensure non-empty lines
                # Format the current time as mm:ss
                formatted_time = str(current_time).split('.')[0][2:]

                # Write the line to the file with timestamp
                output.write(f"{formatted_time} - {line}\n")
                output.flush() # Flush the output buffer to write the line to the file
                print(f"Added: {formatted_time} - {line}")

                # Generate a random time interval between 3 to 6 seconds
                interval = random.randint(3, 6)

                # Update the current time
                current_time += timedelta(seconds=interval)

                # Sleep for the generated interval
                time.sleep(interval)

# Simulate updating the text file
simulate_lecture_updates("lecture_input.txt", "simulated_lecture_output.txt")
