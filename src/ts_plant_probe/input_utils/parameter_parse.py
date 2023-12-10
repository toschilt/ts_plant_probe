import rospy
from rospy.exceptions import ROSException

class ParameterParse():
    """
    Creates a parser for roslaunch parameters.
    """

    def __init__(self) -> None:
        self.parameters = self.get_all_parameters()

    def get_all_parameters_names(self) -> list:
        """
        Gets all parameter names from the parameter server.

        Returns:
            A list containing all parameter names.
        """
        return rospy.get_param_names()
        
    def get_all_parameters(self) -> dict:
        """
        Gets all parameters from the parameter server.

        Returns:
            A dictionary containing all parameters.
        """
        parameters = {}
        for parameter_name in self.get_all_parameters_names():
            # Remove the leading slash
            _param_name = parameter_name[1:] 
            parameters[_param_name] = rospy.get_param(_param_name).format(**parameters)
        return parameters

    