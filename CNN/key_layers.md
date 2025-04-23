Layer | What it does | Key Params
--- | --- | ---
Conv2D | Learns filters (features) like edges, curves | filters, kernel_size, strides
ReLU | Introduces non-linearity | None
MaxPooling2D | Downsamples (reduces spatial size) | pool_size, strides
BatchNorm | Stabilizes learning by normalizing activations | axis
Dropout | Randomly deactivates neurons during training | rate
Flatten | Converts 2D features to 1D | None
Dense | Fully connected layers for classification | units, activation