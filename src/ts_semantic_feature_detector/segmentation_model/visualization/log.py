"""
"""

import matplotlib.pyplot as plt
import numpy as np
import json

class MetricsVisualizer:
    """
    Implements several visualizations for the segmentation model metrics.

    Useful reference for the meaning of the AP and AR values:
    https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2

    Attributes:
        train_metrics: a dictionary containing the training metrics.
        validation_metrics: a dictionary containing the validation metrics.

    #TODO: some methods need documentation.
    """

    def __init__(
        self,
        train_log_path: str,
        validation_log_path: str
    ):
        """
        Initializes the visualizer.

        Args:
            train_log_path: a string containing the path to the .json
                file containing the training metrics.
            validation_log_path: a string containing the path to the 
                .json file containing the validation metrics.
        """

        self.train_metrics = self._load_train_log(train_log_path)
        self.validation_metrics = self._load_validation_log(validation_log_path)

    def _load_train_log(
        self,
        train_log_path: str
    ):
        """
        Read the training log file and load the metrics.

        Args:
            train_log_path: a string containing the path to the .json
                file containing the training metrics.

        Returns:
            a dictionary containing each training metric as the key. Each
            metric is represented as a list, ordered by the epoch that the
            value was generated.
        """
        training_data = None
        with open(train_log_path, "r") as input:
            training_data = json.load(input)

        epochs = []
        lr = []
        loss = []
        loss_classifier = []
        loss_box_reg = []
        loss_mask = []
        loss_objectness = []
        loss_rpn_box_reg = []

        for epoch in training_data:
            epochs.append(int(epoch))
            epoch_data = training_data[epoch]
            
            lr.append(epoch_data['lr'])
            loss.append(epoch_data['loss'])
            loss_classifier.append(epoch_data['loss_classifier'])
            loss_box_reg.append(epoch_data['loss_box_reg'])
            loss_mask.append(epoch_data['loss_mask'])
            loss_objectness.append(epoch_data['loss_objectness'])
            loss_rpn_box_reg.append(epoch_data['loss_rpn_box_reg'])

        return {'epochs': epochs, 
            'lr':lr,
            'loss':loss,
            'loss_classifier':loss_classifier,
            'loss_box_reg':loss_box_reg,
            'loss_mask':loss_mask,
            'loss_objectness': loss_objectness,
            'loss_rpn_box_reg': loss_rpn_box_reg}
    
    def _load_validation_log(
        self,
        validation_log_path: str
    ):
        """
        Read the validation log file and load the metrics.

        Args:
            validation_log_path: a string containing the path to the .json
                file containing the validation metrics.

        Returns:
            a dictionary containing each validation metric as the key. Each
            metric is represented as a Numpy array, ordered by the epoch that
            the value was generated.
        """

        testing_data = None
        with open(validation_log_path, "r") as input:
            testing_data = json.load(input)

        epochs = []
        bbox = []
        segm = []

        for epoch in testing_data:
            epochs.append(int(epoch))
            bbox.append(testing_data[epoch]['bbox'])
            segm.append(testing_data[epoch]['segm'])

        return {'epochs': epochs,
                'bbox': np.array(bbox),
                'segm': np.array(segm)}
    
    def plot_training_loss_metrics_separated(self):
        plt.figure()
        plt.suptitle('Training losses', fontsize=18)
        plt.subplots_adjust(top=0.935,
                            bottom=0.06,
                            left=0.04,
                            right=0.96,
                            hspace=0.4,
                            wspace=0.155)
                            
        loss_metrics = list(self.train_metrics.keys())[2:]
        for i, loss_metric in zip(range(len(loss_metrics)), loss_metrics):
            plt.subplot(3, 2, i + 1)
            plt.plot(self.train_metrics['epochs'], self.train_metrics[loss_metric])
            
            if loss_metric == 'loss':
                plt.title('loss_sum')
            else:
                plt.title(loss_metric)

            plt.xlabel('Epochs')
            plt.ylabel('Loss')

        plt.show()

    def plot_training_loss_metrics_together(self):
        plt.figure()
        plt.suptitle('Training losses', fontsize=18)
        
        loss_metrics = list(self.train_metrics.keys())[2:]
        for loss_metric in loss_metrics:
            plt.plot(self.train_metrics['epochs'], self.train_metrics[loss_metric], label=loss_metric)
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_training_loss_and_lr(self):
        fig, ax1 = plt.subplots()
        plt.suptitle('Training losses and learning rate', fontsize=18)
        
        loss_metrics = list(self.train_metrics.keys())[2:]
        for loss_metric in loss_metrics:
            ax1.plot(self.train_metrics['epochs'], self.train_metrics[loss_metric], label=loss_metric)

        ax1.legend()
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epochs')
        
        ax2=ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(self.train_metrics['epochs'], self.train_metrics['lr'], 'b', label='lr')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc=0)
        
        plt.show()

    def plot_learning_rate(self):
        plt.figure()
        plt.suptitle('Learning rate', fontsize=18)
        plt.plot(self.train_metrics['epochs'], self.train_metrics['lr'])
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.show()

    #type = 'bbox' or 'segm'
    def plot_testing_metrics(self, type):
        plt.figure()
        if type == 'bbox':
            plt.suptitle('Bounding box testing metrics', fontsize=18)
        elif type == 'segm':
            plt.suptitle('Segmentation mask testing metrics', fontsize=18)
        
        plt.subplots_adjust(top=0.935,
                            bottom=0.06,
                            left=0.04,
                            right=0.96,
                            hspace=0.38,
                            wspace=0.155)

        plt.subplot(3, 2, 1)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,0])
        plt.title('AP at IoU=.50:.05:.95')
        plt.xlabel('Epochs')
        plt.ylabel('Average Precision')

        plt.subplot(3, 2, 3)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,1])
        plt.title('AP at IoU=.50')
        plt.xlabel('Epochs')
        plt.ylabel('Average Precision')

        plt.subplot(3, 2, 5)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,2])
        plt.title('AP at IoU=.75')
        plt.xlabel('Epochs')
        plt.ylabel('Average Precision')

        plt.subplot(3, 2, 2)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,6])
        plt.title('AR given 1 detection per image')
        plt.xlabel('Epochs')
        plt.ylabel('Average Recall')

        plt.subplot(3, 2, 4)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,7])
        plt.title('AR given 10 detections per image')
        plt.xlabel('Epochs')
        plt.ylabel('Average Recall')

        plt.subplot(3, 2, 6)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,8])
        plt.title('AR given 100 detections per image')
        plt.xlabel('Epochs')
        plt.ylabel('Average Recall')
        
        plt.show()

    def plot_testing_metrics_comparatively(self, type):
        plt.figure()
        
        plt.subplots_adjust(top=0.895,
                            bottom=0.065,
                            left=0.125,
                            right=0.9,
                            hspace=0.24,
                            wspace=0.2)

        if type == 'bbox':
            plt.suptitle('Bounding box testing metrics', fontsize=18)
        elif type == 'segm':
            plt.suptitle('Segmentation mask testing metrics', fontsize=18)

        plt.subplot(2, 1, 1)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,0], label='AP at IoU=.50:.05:.95')
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,1], label='AP at IoU=.50')
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,2], label='AP at IoU=.75')
        plt.title('Average Precision (AP)')
        plt.xlabel('Epochs')
        plt.ylabel('Average Precision')
        plt.legend(loc='lower left')

        plt.subplot(2, 1, 2)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,6], label='AR given 1 detection per image')
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,7], label='AR given 10 detections per image')
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics[type][:,8], label='AR given 100 detections per image')
        plt.title('Average Recall (RP)')
        plt.xlabel('Epochs')
        plt.ylabel('Average Recall')
        plt.legend(loc='lower left')
        
        plt.show()

    def plot_relevant_metrics(self):
        plt.figure()

        plt.subplot(1, 2, 1)
        loss_metrics = list(self.train_metrics.keys())[2:]
        for loss_metric in loss_metrics:
            plt.plot(self.train_metrics['epochs'], self.train_metrics[loss_metric], label=loss_metric)
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics['bbox'][:,0], label='AP at IoU=.50:.05:.95')
        plt.plot(self.validation_metrics['epochs'], self.validation_metrics['bbox'][:,8], label='AR given 100 detections per image')
        plt.title('Validation metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    metrics = MetricsVisualizer(
        '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/log/train_log.json',
        '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/log/validation_log.json'
    )
    metrics.plot_relevant_metrics()
    metrics.plot_training_loss_metrics_together()
    metrics.plot_testing_metrics_comparatively('segm')