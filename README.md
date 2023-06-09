# Monocular Calibration


## Extract intrinsic and extrinsic properties of a monocular

This camera calibration was done by using *OpenCV*. This is just a simple test done by using random chessboard images found on Internet.

The output displayed on the terminal is saved in *result.txt*. The script also calculates the **re-projection error**.

## Validating Estimates vs Ground Truth

Run the simulation:

`./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml ~/Datasets/EuRoC/MH01 ./Examples/Monocular/EuRoC_TimeStamps/MH01.txt dataset-MH01_mono`


Then, compare with Ground Truth

`python2.7 evaluation/evaluate_ate_scale.py --verbose evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt f_dataset-MH01_mono.txt --plot MH01_mono.pdf`

With the *--verbose argument*, it gives this kind of output:

<pre>
compared_pose_pairs 3638 pairs
absolute_translational_error.rmse 3.589711 m
absolute_translational_error.mean 3.303712 m
absolute_translational_error.median 3.697012 m
absolute_translational_error.std 1.404103 m
absolute_translational_error.min 0.776417 m
absolute_translational_error.max 6.239657 m
max idx: 2618
</pre>

The plot of the trajectory is saved in *MH01_mono.pdf*


## Bibliography

- OpenCV documentation, https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Images of chessboards, https://www.me.psu.edu/brennan/ME545/2012/FinalProjects/Mangus_3DCameraCalibration/Calibration_Images.html
