unset PYTHONPATH
pyenv activate ts_plant_probe
source /home/ltoschi/Documents/LabRoM/plant_by_plant_detection/ts_ws/devel/setup.sh

folder_path="/home/ltoschi/Documents/LabRoM/plant_by_plant_detection/ts_ws/src/ts_plant_probe/src/"

if [ -d "$folder_path" ]; then
    export PYTHONPATH="$folder_path:$PYTHONPATH"
    echo "Added '$folder_path' to PYTHONPATH."
else
    echo "The specified path '$folder_path' is not a directory."
fi