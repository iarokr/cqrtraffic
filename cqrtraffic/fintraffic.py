# Import dependencies
from pystoned import wCQER
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import enum
import numpy as np

from cqrtraffic.utils import load, process, constant

DEFAULT_TAU_LIST = [0.5]


class Representations(enum.Enum):
    agg = 1
    bag = 2
    bag_w = 3


class FintrafficTMS:
    """
    A class to represent data for a Traffic Measurement Station from several days.
    """

    def __init__(
        self,
        tms_id: int,
        days_list: list,
        direction: int,
        tau_list: list = DEFAULT_TAU_LIST,
        hour_from: int = 6,
        hour_to: int = 20,
    ):
        self.tms_id = tms_id

        # Date and time information
        self.days_list = days_list
        self.hour_from = hour_from
        self.hour_to = hour_to

        # Vehicle positioning
        self.direction = direction

        # Quantiles to model
        self.tau_list = tau_list

    def load_raw_data(self):
        self.raw_data = load.read_several_reports(
            tms_id=self.tms_id,
            year_day_list=self.days_list,
            direction=self.direction,
        )

    def aggregate(self, aggregation_time_period: int = constant.DEF_AGG_TIME_PER):
        self.agg_data = process.aggregate(
            data=self.raw_data,
            direction=self.direction,
            aggregation_time_period=aggregation_time_period,
        )
        self.agg_time_period = aggregation_time_period
        self.agg_flow = self.agg_data["flow"]
        self.agg_density = self.agg_data["density"]

    def bagging(self, gridsize_x: int = 70, gridsize_y: int = 400):
        self.bag_data = process.bagging(self.agg_data, gridsize_x, gridsize_y)
        self.gridsize_x = gridsize_x
        self.gridsize_y = gridsize_y
        self.bag_density = self.bag_data["centroid_density"]
        self.bag_flow = self.bag_data["centroid_flow"]
        self.weight = self.bag_data["weight"]

    def weighted_model(self, tau_list: list = None, email: str = None):
        self.bagged_model = []
        self.time_weighted = []
        if tau_list is None:
            tau_list = self.tau_list
        for tau in tau_list:
            start_time = time.perf_counter()
            model = wCQER.wCQR(
                y=self.bag_data.centroid_flow,
                x=self.bag_data.centroid_density,
                w=self.weight,
                tau=tau,
            )
            model.__model__.beta.setlb(None)
            if email is not None:
                model.optimize(email)
            else:
                model.optimize()
            self.bagged_model.append(model)
            end_time = time.perf_counter()
            self.time_weighted.append(end_time - start_time)

    def plot_data(self, representation: Representations):
        if isinstance(representation, Representations):
            plt.clf()
            figure(figsize=(10, 10), dpi=400)

            if representation == Representations.agg:
                plt.scatter(
                    self.agg_density,
                    self.agg_flow,
                    marker="x",
                    c="black",
                    label="Aggregated data",
                )
            elif representation == Representations.bag:
                plt.scatter(
                    self.bag_density,
                    self.bag_flow,
                    marker="x",
                    c="black",
                    label="Bagged data",
                )
            elif representation == Representations.bag_w:
                plt.scatter(
                    self.bag_density,
                    self.bag_flow,
                    marker="o",
                    c="black",
                    s=self.weight * 10000,
                    label="Bagged data with weighted representation",
                )

            plt.xlabel("Density [veh/km]")
            plt.ylabel("Flow [veh/h]")
            plt.legend()
        else:
            raise Exception(f"Representation {representation} does not exist")

    def plot_model(self, data_representation: Representations):
        if isinstance(data_representation, Representations):
            plt.clf()
            figure(figsize=(10, 10), dpi=400)

            if data_representation == Representations.agg:
                plt.scatter(
                    self.agg_density,
                    self.agg_flow,
                    marker="x",
                    c="black",
                    label="Aggregated data",
                )
            elif data_representation == Representations.bag:
                plt.scatter(
                    self.bag_density,
                    self.bag_flow,
                    marker="x",
                    c="black",
                    label="Bagged data",
                )
            elif data_representation == Representations.bag_w:
                plt.scatter(
                    self.bag_density,
                    self.bag_flow,
                    marker="o",
                    c="black",
                    s=self.weight * 10000,
                    label="Bagged data with weighted representation",
                )

            cmap = plt.get_cmap("plasma")
            slicedCM = cmap(np.linspace(0, 1, len(self.bagged_model)))
            color_index = 0
            for model in self.bagged_model:
                x = np.array(model.x).flatten()
                y = np.array(model.y).T
                yhat = np.array(model.get_frontier()).T

                data = (np.stack([x, y, yhat])).T

                # sort
                data = data[np.argsort(data[:, 0])].T
                x, y, f = data[0], data[1], data[2]
                plt.plot(
                    x,
                    f,
                    c=slicedCM[color_index],
                    label="tau=" + str(self.tau_list[color_index]),
                )
                color_index += 1

            plt.xlabel("Density [veh/km]")
            plt.ylabel("Flow [veh/h]")
            plt.legend()
        else:
            raise Exception(f"Representation {data_representation} does not exist")

    def make_model(
        self,
        aggregation_time_period: int = constant.DEF_AGG_TIME_PER,
        gridsize_x: int = 70,
        gridsize_y: int = 400,
        tau_list: list = None,
        email: str = None,
    ):
        if tau_list is None:
            tau_list = self.tau_list
        self.load_raw_data()
        self.aggregate(aggregation_time_period=aggregation_time_period)
        self.bagging(gridsize_x=gridsize_x, gridsize_y=gridsize_y)
        self.weighted_model(tau_list=tau_list, email=email)
