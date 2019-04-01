
class MetricImagesContainer:

    def __init__(self):
        self.images = {}


class Result:

    def __init__(self, team, aoi, data):
        """

        :param team: Team name i.e. "ARA"
        :param aoi:  AOI name i.e. "D4"
        :param data: Json data dictionary
        """
        self.team = team
        self.aoi = aoi
        self.results = data


class SummarizedMetrics:

    def __init__(self, team, summarized_results):
        self.team = team
        self.summarized_results = summarized_results
