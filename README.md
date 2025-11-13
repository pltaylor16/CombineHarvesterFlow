# CombineHarvesterFlow

## Pip installation instructions

To install CombineHarvesterFlow on a CPU machine run:

    pip install CombineHarvesterFlow

There are issues with pip-installed versions of jax on some machines. If this is an issue, you can also follow the alternative install instructions below.

On GPU machines you will then have to reinstall jax with your correct CUDA version 

## Alternative installation instructions

CombineHarvesterFlow uses [jax](https://github.com/google/jax) and [flowjax](https://github.com/danielward27/flowjax). The following installation instructions will automatically install both CombineHarvesterFlow and all required packages:

    conda create -n jax python==3.10
    conda activate jax

Follow the jax install instructions [here](https://jax.readthedocs.io/en/latest/installation.html) for CPU or GPU depending on your local hardware. These instructions can fail on some machines in which case we recommend the conda-forge installation:
    
    conda install -c conda-forge jaxlib
    conda install -c conda-forge jax
 
Once jax is installed run: 

    git clone https://github.com/pltaylor16/CombineHarvesterFlow.git
    cd CombineHarvesterFlow
    pip install .

Training of the normalizing flows can be slower when using CPUs (this is not normally problematic in low-dimensions e.g. n<7). Therefore we recommend using GPUs, if possible.

## Perlmutter (NERSC) installation instructions

The process for installing CombineHarvesterFlow at NERSC with access to GPUs is slightly more involved. First, we need to install jax following the [NERSC documentation](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#jax):

    module load cudatoolkit/12.2
    module load cudnn/8.9.3_cuda12
    module load python
    # Create a new conda environment
    conda create -n jax python=3.10 pip numpy scipy
    # Activate the environment before using pip to install JAX
    conda activate jax
    # Install a compatible wheel
    pip install --no-cache-dir "jax==0.4.23" "jaxlib[cuda12_cudnn89]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

See the [NERSC documentation](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#jax) for more details on how to find a working version of jax. After that, install CombineHarvesterFlow:
    
    git clone https://github.com/pltaylor16/CombineHarvesterFlow.git
    cd CombineHarvesterFlow
    pip install .

If you want to run CombineHarvesterFlow in a notebook, you will need to set up a helper script that automatically loads the Cuda modules. Detailed instructions for this can be found [here](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-customize-a-kernel-with-a-helper-shell-script) and are summarized below. First setup the Jupyter kernel:

    pip install ipykernel
    python -m ipykernel install --user --name jax --display-name Jax

Then create a helper script for the jupyter kernel:

    touch $HOME/.local/share/jupyter/kernels/jax/kernel-helper.sh

After that, add the following lines to that new script:

    #!/bin/bash
    module load cudatoolkit/12.2
    module load cudnn/8.9.3_cuda12
    module load python
    conda activate jax
    exec "$@"

and make the script executable:

    chmod u+x $HOME/.local/share/jupyter/kernels/jax/kernel-helper.sh

Finally, modify the kernel json file here: `$HOME/.local/share/jupyter/kernels/jax/kernel.json`, to automatically run the helper script when starting the kernel (if opening the file from the JupyterHub interface, right-click on it and select Open With -> Editor). The file should look something like this (where the `"{resource_dir}/kernel-helper.sh"` line is new):

    {
     "argv": [
      "{resource_dir}/kernel-helper.sh",
      "python",
      "-m",
      "ipykernel_launcher",
      "-f",
      "{connection_file}"
     ],
     "display_name": "Jax",
     "language": "python",
     "metadata": {
      "debugger": true
     }
    }

For detailed instructions, see the [NERSC jax documentation](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#jax) and the [NERSC kernel customization documentation](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-customize-a-kernel-with-a-helper-shell-script).
