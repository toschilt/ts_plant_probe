# TerraSentia Plant Probe

This repository contains a ROS package that implements a semantic feature detector for cornfields. It uses data gathered with TerraSentia, a autonomous robot developed by the University of Illinois (Urbana-Champaign). Research supported by grant #2022/13040-0, São Paulo Research Foundation (FAPESP).

## Setup

### Requirements

- ROS Noetic
- Python 3.8.10

### Installation

#### Installing ROS Noetic

To install ROS Noetic, follow the instructions in the [ROS Noetic installation page](http://wiki.ros.org/noetic/Installation). 

When installing ROS, many Python packages will be also installed. As you may want to install them for just this project, we recommend creating and activating a Python virtual environment before installing ROS. See the next section for instructions.

If you want to use ROS on other projects, you can install it globally. In this case, you do not need to create a Python virtual environment before installing ROS. We still recommend creating a virtual environment for the specific Python dependencies of this project. See the next section for instructions.
#### Creating a Python Virtual Environment

To make sure that the dependencies are installed in a isolated environment, we recommend creating a Python virtual environment. We are using pyenv-virtualenv to manage our virtual environments. To install it, follow the instructions in the [pyenv-virtualenv repository](https://github.com/pyenv/pyenv-virtualenv).

Before creating the environment, we suggest unsetting the `PYTHONPATH` environment variable, if it is set. This is because the `PYTHONPATH` variable may interfere with the virtual environment. To unset it, run:

```bash
unset PYTHONPATH
```

After installing pyenv-virtualenv, create a new virtual environment with Python 3.8.10.

```bash
pyenv install 3.8.10
pyenv virtualenv 3.8.10 ts_plant_probe
```

After that, activate the virtual environment:

```bash
pyenv activate ts_plant_probe
```

#### Installing Python Dependencies

To install the Python dependencies, run:

```bash
pip install pip --upgrade
pip install -r requirements.txt
```

#### Creating the ROS Workspace and compiling

To create the ROS workspace with this package inside it, run:

```bash
mkdir -p ts_ws/src
cd ts_ws/src
```

Then, clone this repository inside the `src` folder:

```bash
git clone https://github.com/toschilt/ts_plant_probe.git
```

If you installed ROS globally and it is following this tutorial in the sequence, you need to set the `PYTHONPATH` environment variable to reinclude the system packages. To do that, you can restart the terminal:
```bash
pyenv deactivate
exec $SHELL
```

After that, compile the workspace:

```bash
cd ..
catkin_make
```

## Running

To run the project, follow the steps below.

1. Unset the `PYTHONPATH` environment variable
```bash
unset PYTHONPATH
```

2. Activate the Python virtual environment
```bash
pyenv activate ts_plant_probe
```

3. Source the ROS workspace
```bash
source ts_ws/devel/setup.bash
```

4. Add the `ts_plant_probe` package to the `PYTHONPATH` environment variable (for testing). Substitute `{PATH_TO_TS_WS}` with the path to the `ts_ws` folder.
```bash
export PYTHONPATH=$PYTHONPATH:{PATH_TO_TS_WS}/ts_ws/src/ts_plant_probe
```

5. Run the `demo_node` node

In other terminal (do not prior configuration):
```bash
roscore
```

In the configured terminal:

```bash
rosrun ts_plant_probe demo_node.py
```

## Testing

We are using pytest to run the tests. To run the tests, follow the steps 1 to 4 in the [Running](#running) section. Then, run:

```bash
cd ts_ws/src/ts_plant_probe
pytest tests 
```