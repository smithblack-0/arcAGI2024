def make_callbacks(num_workers):
    callbacks = []
    for i in range(num_workers):
        def callback():
            print(f"Callback for worker {i}")
        callbacks.append(callback)
    return callbacks

# Generate callbacks
callbacks = make_callbacks(3)
for callback in callbacks:
    callback()