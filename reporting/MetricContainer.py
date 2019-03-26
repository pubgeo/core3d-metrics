
class MetricImagesContainer:

    def __init__(self):
        self.images = {}


class Result:

    def __init__(self, team, test, aoi, data):
        """

        :param team: Team name i.e. "ARA"
        :param test: Test name i.e. "Self_Test_2019"
        :param aoi:  AOI name i.e. "D4"
        :param data: Json data dictionary
        """
        self.team = team
        self.test = test
        self.aoi = aoi
        self.results = data

