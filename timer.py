# file that contains a simple timer function to track how long the process takes

import time

def start_timer():
    """
    Starts a timer and returns the start time.

    Returns:
        float: The start time.
    """
    return time.time()

def end_timer(start_time):
    """
    Ends the timer and prints the elapsed time.

    Args:
        start_time (float): The start time from when the timer was started.
    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
