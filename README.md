# Micro-event-detection-from-3-axis-accelerometer
Overview

This project is a small synthetic prototype that simulates 3-axis accelerometer signals and demonstrates a basic sensor-data processing pipeline — generation, feature extraction, and classification.
The goal is to model micro-events (short bursts of motion) and distinguish them from idle periods using simple statistical features and a lightweight machine-learning model.

No physical IoT hardware is used here; all signals are mathematically generated to imitate realistic accelerometer patterns.
I used a low frequency sine wave as a base, added a pulse by using a gausian at random, and added sensor noises. I also made sure all the axes show slightly different wave forms.
This work was done as a learning exercise to understand how signal windows can be processed and classified before moving to real-world IoT sensor data.

Uses RandomForestClassifier from scikit-learn.
Data split 75 % train and 25 % test with stratified sampling.
