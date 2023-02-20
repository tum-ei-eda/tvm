#!/bin/bash  
#python3 run.py aww --kernel-layout HWIO --profile --verbose
#python3 run.py aww --data-layout NHWC --kernel-layout ? --arch rv32gpc --profile (--verbose)

cd ..
for model in 'aww' 'vww' 'resnet' 'toycar'

do
    echo §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    echo model: $model

    for data_layout in 'NHWC' 'NCHW'
    do
        echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        echo data layout:$data_layout

        # for kernel in 'HWIO' 'OIHW'
        for kernel in 'default'
        do
            echo ----------------------------------------------------
            echo kernel layout:$kernel
            python3 run.py $model --data-layout $data_layout --kernel-layout $kernel  --profile
        done
    done
done
