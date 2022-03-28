import cv2 as cv


class MetricsContainer:

    def __init__(self):
        """
        Constructor for metrics container
        """
        self.name = None
        self.TP = None
        self.FP = None
        self.FN = None
        self.ignored_ground_truth_ids = []
        self.ignored_performer_ids = []
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.matched_gt_ids = []
        self.unmatched_gt_ids = []
        self.matched_performer_ids = []
        self.unmatched_performer_ids = []
        self.average_area_difference = None
        self.average_area_ratio = None
        self.stoplight_chart = None
        self.iou_per_gt_building = {}

    def show_stoplight_chart(self):
        """
        Brings up a opencv window that displays the stoplight chart and waits for keypress to close
        :return:
        """
        if self.stoplight_chart is None:
            print("Stoplight chart does not exist. Please generate the stoplight chart first!")
            return
        else:
            cv.namedWindow('stoplight', cv.WINDOW_NORMAL)
            cv.imshow('stoplight', self.stoplight_chart)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def set_values(self, tp, fp, fn, ignored_ground_truth_ids, ignored_performer_ids, precision, recall, f1_score,
                   matched_gt_ids, unmatched_gt_ids, matched_performer_ids, unmatched_performer_ids,
                   average_area_difference, average_area_ratio, stoplight_chart, iou_per_gt_building):
        """
        Setter method that simply let you set object variables in one function instead of individually
        :param tp: True positive value
        :param fp: False positive value
        :param fn: False negative value
        :param ignored_ground_truth_ids: list of ignored ground truth IDs
        :param ignored_performer_ids: list of ignored performer IDs
        :param precision: TP / (TP + FP)
        :param recall: TP / (TP + FN)
        :param f1_score: 2 * precision * recall / (precision + recall)
        :param matched_gt_ids: list of matched ground truth IDs
        :param unmatched_gt_ids: list of unmatched ground truth IDs, or list of false negative IDs
        :param matched_performer_ids: list of matched performer IDs
        :param unmatched_performer_ids: list of unmatched performer IDs, or list of false positives
        :param average_area_difference: mean of all matched area differences
        :param average_area_ratio: mean of all matched area ratios
        :param stoplight_chart: numpy array of stoplight chart raster
        :param iou_per_gt_building: dictionary of iou and centroid in image per gt building
        :return:
        """
        self.TP = tp
        self.FP = fp
        self.FN = fn
        self.ignored_ground_truth_ids = ignored_ground_truth_ids
        self.ignored_performer_ids = ignored_performer_ids
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.matched_gt_ids = matched_gt_ids
        self.unmatched_gt_ids = unmatched_gt_ids
        self.matched_performer_ids = matched_performer_ids
        self.unmatched_performer_ids = unmatched_performer_ids
        self.average_area_difference = average_area_difference
        self.average_area_ratio = average_area_ratio
        self.stoplight_chart = stoplight_chart
        self.iou_per_gt_building = iou_per_gt_building

    def show_metrics(self, suppress_lists=True):
        """
        Prints out metrics stored in object
        :return:
        """
        print("Printing Results...")
        print("TP:" + repr(self.TP))
        print("FP: " + repr(self.FP))
        print("FN: " + repr(self.FN))
        print("Number of ignored GT: " + repr(len(self.ignored_ground_truth_ids)))
        print("Ignored Perf: " + repr(len(self.ignored_performer_ids)))
        print("Precision: " + repr(self.precision))
        print("Recall: " + repr(self.recall))
        print("F1-Score: " + repr(self.f1_score))
        if suppress_lists is False:
            print("Ignored GT Indices: " + repr(self.ignored_ground_truth_ids))
            print("Unmatched GT Indices: " + repr(self.unmatched_gt_ids))
            print("Ignored Perf Indices: " + repr(self.ignored_performer_ids))
            print("Matched Perf Indices: " + repr(self.matched_performer_ids))
            print("Unmatched Perf Indices: " + repr(self.unmatched_performer_ids))
        print("Average Area Difference: " + repr(self.average_area_difference))
        print("Average Area Ratio: " + repr(self.average_area_ratio))