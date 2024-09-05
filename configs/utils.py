from datetime import datetime
import inspect

def log_current_time():
    """
    Print and log the current date and time along with the name of the function it is executed from.
    """
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Get the name of the calling function
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name if caller_frame else "Unknown"

    # Prepare the log entry
    log_entry = f"{formatted_time} - {caller_name}"

    # Print to console
    print(f"Log entry: {log_entry}")

    # Append to file
    with open('time_log.txt', 'a') as file:
        file.write(f"{log_entry}\n")