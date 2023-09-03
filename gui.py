import tkinter as tk
from threading import Thread
from train import train_model, load_model_for_training
from utils import get_device

# Global flag to signal the training thread to stop
training_stop_flag = False

# Define a function to start training in a separate thread
def start_training_thread(args, train_button, stop_button):
    global training_stop_flag  # Use the global flag
    training_stop_flag = False  # Reset the flag
    train_button.config(state=tk.DISABLED)  # Disable the "Train" button
    stop_button.config(state=tk.NORMAL)  # Enable the "Stop Training" button

    thread = Thread(target=start_training, args=(args, train_button, stop_button))
    thread.start()

# Define a function to stop training
def stop_training(train_button, stop_button):
    global training_stop_flag
    training_stop_flag = True
    stop_button.config(state=tk.DISABLED)  # Disable the "Stop Training" button

# Define a function to start training when the "Train" button is clicked
def start_training(args, train_button, stop_button):
    global training_stop_flag
    print("Training started...")
    
    # Reset the flag at the start of training
    training_stop_flag = False

    train_model(*args)

    if training_stop_flag:
        print("Training stopped by user.")
    else:
        print("Training finished.")  # Add a message to indicate training completion
    train_button.config(state=tk.NORMAL)  # Re-enable the "Train" button
    stop_button.config(state=tk.DISABLED)  # Disable the "Stop Training" button

# Create the GUI
def create_gui(args):
    # Create the main application window
    root = tk.Tk()
    root.title("Super Mario Bros RL Bot")

    # Add a Text widget to display hyperparameters
    hyperparameters_text = tk.Text(root, height=10, width=40)
    hyperparameters_text.grid(row=0, column=0, padx=10, pady=10)
    hyperparameters_text.insert(tk.END, "Hyperparameters:\n")
    hyperparameters_text.insert(tk.END, f"Number of CPUs: {args.num_cpu}\n")
    hyperparameters_text.insert(tk.END, f"Skip: {args.skip}\n")
    hyperparameters_text.insert(tk.END, f"Learning rate: {args.learning_rate}\n")
    hyperparameters_text.insert(tk.END, f"Log directory: {args.log_dir}\n")
    hyperparameters_text.insert(tk.END, f"Environment id: {args.env_id}\n")

    # Add a "Train" button
    train_button = tk.Button(root, text="Train", command=lambda: start_training_thread(
        (load_model_for_training(args.model, args.env_id, args.skip, args.num_cpu, get_device(), args.learning_rate),
         args.log_dir, args.learning_rate),
        train_button, stop_button
    ))
    train_button.grid(row=0, column=1, padx=10, pady=10)

    # Add a "Stop Training" button
    stop_button = tk.Button(root, text="Stop Training", command=lambda: stop_training(train_button, stop_button), state=tk.DISABLED)
    stop_button.grid(row=0, column=2, padx=10, pady=10)

    # Start the Tkinter main loop
    root.mainloop()
