# CTRL Repo

This project uses [Placo](https://placo.readthedocs.io/) for robot control and kinematics. [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html) is used as the physics simulator.

This project is still in its early stages, and the documentation will be updated as development progresses.

Follow the steps below to set up a reproducible environment.

## Setup Instructions
1. Clone the repository along with its submodules:
   ```bash
   git clone --recurse-submodules <repository_url>
   cd <repository_directory>
   ```

2. Create and activate the conda environment:
   ```bash
    conda env create -f environment.yml
    conda activate robot-env
    ```

3. Build placo from source following the instructions [here](https://placo.readthedocs.io/en/latest/basics/installation_source.html)

4. Don't forget to add the placo bindings to your PYTHONPATH. You can do this by adding the following line to your `.bashrc` or `.zshrc` file:
   ```bash
   export PYTHONPATH=$PYTHONPATH:<ctrl repo path>/extern/placo/build/lib/python3-X/site-packages
   ```