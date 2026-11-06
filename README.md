# Intelligent Pesticide Sprinkling System Using CNN

A computer-vision prototype that classifies plant leaf images and
uses a Python decision layer to control a water pump through an Arduino Uno.

## Planned pipeline

Phone camera -> OpenCV -> CNN inference -> confidence and stability checks
-> USB serial -> Arduino Uno -> relay -> water pump

## Safety

- Use water for prototype testing, not real pesticide.
- The pump must default to OFF.
- Add activation-duration limits and a cooldown.
- Keep water away from the electronics.
- Never power a pump directly from an Arduino GPIO pin.

## Development stages

1. Dataset inspection and leakage-resistant splitting
2. Custom CNN baseline
3. MobileNetV2 transfer learning
4. Model evaluation and Grad-CAM
5. Phone-camera inference
6. Arduino integration with a water pump
7. Tests, documentation, and demonstration
